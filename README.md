## 1. Giới thiệu

Dự án này được thiết kế để:
*   Minh họa các bước tiền xử lý dữ liệu cần thiết như xử lý giá trị thiếu, mã hóa biến phân loại, chuẩn hóa và phát hiện điểm ngoại lai.
*   Thực hiện phân tích PCA từ đầu (from scratch).
*   Áp dụng quy trình trên một tập dữ liệu thực tế để giảm chiều và phân tích các metric liên quan.
*   **Link github:** https://github.com/Socciu007/pca.git
## 2. Cấu trúc Dự án
Dự án có cấu trúc thư mục như sau:
.
├── source/
│ ├── PCA.py
│ ├── PreprocessingDataPipeline.py
│ ├── requirements.txt
│ └── index.py
└── README.md

## 3. Cài đặt

Thực hiện theo các bước sau để cài đặt và thiết lập dự án:
1.  **Tạo môi trường ảo (khuyến nghị):**
  ```bash
  python -m venv venv
  ```

2.  **Kích hoạt môi trường ảo:**
  *   **Trên Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
  *   **Trên macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

3.  **Cài đặt các thư viện cần thiết:**
  ```bash
  pip install -r requirements.txt
  ```
4. **Cách chạy:**
  ```bash
  cd source
  python index.py
  ```

## 4. Nguồn dữ liệu

Dữ liệu được sử dụng trong dự án này là tập dữ liệu **"Estimation of Obesity Levels Based On Eating Habits and Physical Condition"** từ UCI Machine Learning Repository (ID: 544).

*   **Link:** [UCI Machine Learning Repository - Obesity Level](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
