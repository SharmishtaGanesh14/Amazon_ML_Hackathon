# ────────── enhanced_multimodal_context_pipeline.py ──────────
import os
import re
import gc
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import cv2
import easyocr

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
tqdm.pandas()

# -------------------------
# EfficientOCR
# -------------------------
class EfficientOCR:
    def __init__(self, lang_list=['en'], max_workers=4, gpu=True):
        self.reader = easyocr.Reader(lang_list, gpu=gpu)
        self.max_workers = max_workers

    def preprocess_image_for_ocr(self, image_path):
        if not os.path.exists(image_path): return None
        img = cv2.imread(image_path)
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced)
        h, w = denoised.shape
        if h < 300:
            scale = 400 / h
            new_w = int(w * scale)
            denoised = cv2.resize(denoised, (new_w, 400), interpolation=cv2.INTER_CUBIC)
        return denoised

    def extract_text_from_image(self, image_path):
        try:
            processed_img = self.preprocess_image_for_ocr(image_path)
            if processed_img is None: return ""
            rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
            results = self.reader.readtext(rgb)
            text = " ".join([res[1] for res in results])
            return text
        except:
            return ""

    def process_images_parallel(self, image_paths):
        ocr_results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {executor.submit(self.extract_text_from_image, p): p for p in image_paths if p is not None}
            for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="OCR Processing"):
                p = future_to_path[future]
                try: ocr_results[p] = future.result()
                except: ocr_results[p] = ""
        return ocr_results

# -------------------------
# Text Cleaning Utilities
# -------------------------
def clean_extract(text):
    text = str(text)
    v = re.search(r"Value:\s*([\d\.]+)", text)
    value = float(v.group(1)) if v else 0
    u = re.search(r"Unit:\s*([A-Za-z ]+)", text)
    unit = u.group(1).strip() if u else None
    text = re.sub(r"Value:\s*[\d\.]+", "", text)
    text = re.sub(r"Unit:\s*[A-Za-z ]+", "", text)
    text = re.sub(r"Item Name:\s*", "", text)
    text = re.sub(r"Bullet Point\s*\d*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\n\r\t\xa0]+", " ", text)
    text = re.sub(r" +", " ", text).strip()
    return pd.Series([text, value, unit])

PACK_PATTERNS = [
    r"(?:pack|case)\s*of\s*(\d+)", r"(\d+)\s*-\s*pack", r"(\d+)\s*count",
    r"\((\d+)\s*count\)", r"\((\d+)\s*ct\)", r"(\d+)\s*ct\b",
    r"\(\s*pack\s*of\s*(\d+)\s*\)", r"\b(\d+)\s*each\b", r"\b(\d+)\s*ea\b",
]
WEIGHT_UNITS = {"oz":28.35,"ounce":28.35,"lb":453.59,"pound":453.59,"g":1.0,"gram":1.0,"kg":1000.0}
VOLUME_UNITS = {"fl oz":29.57,"ml":1.0,"l":1000.0}
COUNT_TOKENS = {"count","ct","each","ea"}

def extract_pack_count(text):
    s = str(text).lower()
    for p in PACK_PATTERNS:
        m = re.search(p, s)
        if m:
            try: return int(m.group(1))
            except: pass
    return 1

def normalize_quantity(value, unit):
    if unit is None or (isinstance(unit,float) and np.isnan(unit)): return 0.0,0.0,0.0
    u = unit.strip().lower().replace("ounces","ounce").replace("lbs","lb").replace("grams","gram")
    if u in WEIGHT_UNITS: return value*WEIGHT_UNITS[u],0.0,0.0
    if u in VOLUME_UNITS: return 0.0,value*VOLUME_UNITS[u],0.0
    if u in COUNT_TOKENS: return 0.0,0.0,value
    return 0.0,0.0,0.0

# -------------------------
# Text Feature Extractor with OCR
# -------------------------
class EnhancedTextFeatureExtractor:
    def __init__(self, sbert_model_path="./all-MiniLM-L6-v2_local",
                 tfidf_max_features=15000, svd_n_components=256):
        os.environ["TRANSFORMERS_OFFLINE"]="1"
        os.environ["HF_DATASETS_OFFLINE"]="1"
        self.sbert_model = SentenceTransformer(sbert_model_path)
        self.tfidf_vec = None
        self.svd_model = None
        self.tfidf_max_features=tfidf_max_features
        self.svd_n_components=svd_n_components
        self.ocr_processor = EfficientOCR(max_workers=4)

    def _get_text_column(self, df):
        for c in df.columns:
            if any(k in c.lower() for k in ["catalog","description","text","content","title"]):
                return c
        raise ValueError("No suitable text column found!")

    def process_with_ocr(self, df, image_dir, fit=True, batch_size=512):
        df = df.copy()
        text_col=self._get_text_column(df)
        # Prepare image paths
        image_paths = []
        for _, row in df.iterrows():
            link=row.get("image_link","")
            if pd.isna(link) or not isinstance(link,str):
                image_paths.append(None)
            else:
                fname=os.path.basename(link.split("?")[0])
                image_paths.append(os.path.join(image_dir,fname) if os.path.exists(os.path.join(image_dir,fname)) else None)
        valid_paths = [p for p in image_paths if p is not None]
        ocr_results=self.ocr_processor.process_images_parallel(valid_paths)
        ocr_texts=[ocr_results.get(p,"") if p else "" for p in image_paths]
        df["ocr_text"]=ocr_texts
        # Combine original text with OCR
        enhanced_texts=[]
        for _, row in df.iterrows():
            original_text=str(row[text_col])
            ocr_text=str(row["ocr_text"])
            enhanced_texts.append(f"{original_text} [OCR_CONTENT] {ocr_text}" if ocr_text.strip() else original_text)
        df["enhanced_text"]=enhanced_texts
        # Numeric features
        df[["clean_text","Value","Unit"]]=df["enhanced_text"].progress_apply(clean_extract)
        df["pack_count"]=df["clean_text"].progress_apply(extract_pack_count)
        qtys=[normalize_quantity(r["Value"],r["Unit"]) for _,r in df.iterrows()]
        df["total_weight_g"], df["total_volume_ml"], df["total_piece_ct"]=zip(*qtys)
        df["ocr_text_length"]=df["ocr_text"].str.len()
        df["has_ocr_content"]=(df["ocr_text_length"]>0).astype(int)
        num_features=df[["pack_count","total_weight_g","total_volume_ml","total_piece_ct","ocr_text_length","has_ocr_content"]].fillna(0).astype(np.float32).values
        num_log=np.log1p(num_features)
        # SBERT embeddings
        texts=df["clean_text"].fillna("").tolist()
        embs=[]
        for i in range(0,len(texts),batch_size):
            batch=texts[i:i+batch_size]
            embs.append(self.sbert_model.encode(batch, show_progress_bar=False, convert_to_numpy=True))
        sbert_feats=np.vstack(embs)
        # TF-IDF + SVD
        if fit:
            self.tfidf_vec=TfidfVectorizer(max_features=self.tfidf_max_features,ngram_range=(1,2),min_df=2,max_df=0.95)
            X_sparse=self.tfidf_vec.fit_transform(df["clean_text"])
            self.svd_model=TruncatedSVD(n_components=self.svd_n_components, random_state=42)
            X_text=self.svd_model.fit_transform(X_sparse)
        else:
            X_sparse=self.tfidf_vec.transform(df["clean_text"])
            X_text=self.svd_model.transform(X_sparse)
        # SBERT embeddings of OCR text
        ocr_texts_list=df["ocr_text"].fillna("").tolist()
        ocr_embs=[]
        for i in range(0,len(ocr_texts_list),batch_size):
            batch=ocr_texts_list[i:i+batch_size]
            ocr_embs.append(self.sbert_model.encode(batch, show_progress_bar=False, convert_to_numpy=True))
        ocr_feats=np.vstack(ocr_embs)
        # Final return
        text_features=np.hstack([X_text,sbert_feats,num_log])
        combined_image_features=np.hstack([ocr_feats])  # img feats concatenated later
        return df["sample_id"].astype(str).values, text_features, combined_image_features

# -------------------------
# Image Feature Extractor
# -------------------------
class OfflineImageFeatureExtractor:
    def __init__(self, device=None, weights_path="./pretrained_weights/efficientnetv2_s_imagenet.pth"):
        self.device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=efficientnet_v2_s(weights=None).to(self.device)
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path,map_location=self.device))
        self.model.classifier=nn.Identity()
        self.model.eval()
        self.transform=Compose([Resize(256),CenterCrop(224),ToTensor(),Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    def preprocess_image(self,img_path):
        try:
            if img_path is None or not os.path.exists(img_path):
                return torch.zeros(3,224,224)
            img=Image.open(img_path).convert("RGB")
            return self.transform(img)
        except:
            return torch.zeros(3,224,224)

    def extract_features(self, paths, batch_size=64, n_workers=8):
        all_feats=[]
        for i in range(0,len(paths),batch_size):
            batch=paths[i:i+batch_size]
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                tens=list(ex.map(self.preprocess_image,batch))
            tb=torch.stack(tens).to(self.device)
            with torch.no_grad():
                feats=self.model(tb).cpu().numpy()
            all_feats.append(feats)
        return np.vstack(all_feats)

    def process(self, df, image_dir, batch_size=128):
        ids=[]
        paths=[]
        for sid, link in zip(df["sample_id"], df.get("image_link", [])):
            ids.append(str(sid))
            if pd.isna(link) or not isinstance(link,str):
                paths.append(None)
            else:
                paths.append(os.path.join(image_dir, os.path.basename(link.split("?")[0])))
        feats=self.extract_features(paths,batch_size=batch_size)
        return np.array(ids), feats

# -------------------------
# Simplified Processor
# -------------------------
class SimplifiedMultimodalProcessor:
    def __init__(self, text_extractor, image_extractor):
        self.text_extractor=text_extractor
        self.image_extractor=image_extractor

    def process_split(self, df, image_dir, fit_text=True):
        ids=df["sample_id"].astype(str).values
        text_ids, text_feats, ocr_feats=self.text_extractor.process_with_ocr(df, image_dir, fit=fit_text)
        _, img_feats=self.image_extractor.process(df, image_dir)
        combined_features=np.hstack([text_feats,img_feats,ocr_feats])
        print(f"✅ Combined features: {combined_features.shape}")
        return ids, combined_features

# -------------------------
# Multimodal Encoder
# -------------------------
class MultimodalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, dropout=0.2):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

# -------------------------
# Main
# -------------------------
def main(dataset_dir="../dataset"):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sbert_path="./all-MiniLM-L6-v2_local"
    weights_path="./efficientnet_v2_s.pth"

    text_extractor=EnhancedTextFeatureExtractor(sbert_model_path=sbert_path)
    image_extractor=OfflineImageFeatureExtractor(weights_path=weights_path)
    processor=SimplifiedMultimodalProcessor(text_extractor,image_extractor)

    for split in ["train","test"]:
        df_path=os.path.join(dataset_dir,f"{split}.csv")
        img_dir=os.path.join(dataset_dir,f"{split}_images")
        if not os.path.exists(df_path): continue
        df=pd.read_csv(df_path)
        fit_text=(split=="train")
        ids, combined_features=processor.process_split(df,img_dir,fit_text=fit_text)

        # Initialize encoder only after knowing feature dim
        encoder=MultimodalEncoder(input_dim=combined_features.shape[1]).to(device)
        with torch.no_grad():
            context_vectors=encoder(torch.from_numpy(combined_features).float().to(device)).cpu().numpy()

        save_path=os.path.join(dataset_dir,f"{split}_context_vectors.npz")
        np.savez_compressed(save_path,sample_ids=ids,context_vectors=context_vectors)
        print(f"💾 Saved {save_path} | Shape: {context_vectors.shape} | Size: {os.path.getsize(save_path)/1024/1024:.1f} MB")

        gc.collect()
        torch.cuda.empty_cache()
    print("🎉 Context vector pipeline complete!")

if __name__=="__main__":
    main()
