# 🛒 QuickCart Warehouse Inventory Stockout Risk Predictor

## 📌 Project Overview

Quick-commerce businesses need to maintain the right inventory levels to avoid two major problems:

- **Stockouts** → Lost sales, empty shelves, and dissatisfied customers
- **Overstocking** → Unnecessary working capital and storage costs

This project develops a **Machine Learning classification system** to predict the daily stockout risk of every **SKU × Store** combination.

Each record is classified into one of three risk categories:

- 🟢 **Safe**
- 🟡 **At-Risk**
- 🔴 **Imminent**

The project uses a relational dataset containing inventory, store, SKU, supplier, and event information and evaluates multiple classification models using a **time-based train-test split**.

---

## 🎯 Business Problem

For every product in every store on every day, the inventory team needs to answer:

> **Is this product likely to run out of stock before the next supplier delivery arrives?**

The model helps identify high-risk SKU-store combinations so that inventory teams can prioritize replenishment decisions.

---

## 📊 Dataset

The project uses a **5-table relational dataset**.

| Dataset | Records | Description |
|---|---:|---|
| `dim_stores.csv` | 12 | Store information |
| `dim_skus.csv` | 60 | Product/SKU information |
| `dim_suppliers.csv` | 15 | Supplier information |
| `dim_events.csv` | 30 | Festival and promotional calendar |
| `fact_inventory_daily.csv` | 21,600 | Daily SKU-store inventory records |

### Dataset Dimensions

- 🏪 **12 Stores**
- 📦 **60 SKUs**
- 🚚 **15 Suppliers**
- 📅 **30 Days**
- 🧾 **21,600 Daily Inventory Records**
- 🏷️ **8 Product Categories**
- 🎯 **3 Stockout Risk Classes**

The modeling grain is:

```text
One row = One SKU × One Store × One Day
