# Data sources

Raw data is never edited. Cleaned versions go in `interim/` and `processed/`. Nothing here is committed to Git.

| Module | Dataset | Put it at | Notes to verify |
|---|---|---|---|
| Crop recommendation | Crop Recommendation dataset (Kaggle, "atharvaingle/crop-recommendation-dataset") | `data/raw/crop/Crop_recommendation.csv` | Very clean and highly separable; treat as a learning baseline. Check units, duplicates, license. No region column, so no regional generalization claims. |
| Mandi prices | Agmarknet or data.gov.in daily prices, or a Kaggle mandi-price dataset as fallback | `data/raw/price/mandi_prices.csv` | Pick ONE commodity and ONE market with >= 2 years of near-daily data. Check units (Rs/quintal), gaps, duplicates. |
| Disease images | PlantVillage (Kaggle) | `data/raw/disease/<class_name>/*.jpg` | Lab-style images, not field conditions. Start with 5 to 6 classes. Beware near-duplicate images across splits. |

Confirm each dataset's license and permitted use on its page before using it, and record it below.

## License log
- Crop recommendation: (https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

- Mandi prices: (https://agmarknet.gov.in/home)

- PlantVillage: (https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)