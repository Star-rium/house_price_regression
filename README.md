# house_price_regression

House price prediction using 3 regression methods for homework yay yippe

### 1. Tiền xử lý dữ liệu

- **Loại bỏ cột nhiễu, thông tin trùng lặp**
  - `date`: Dữ liệu thu thập trong khoảng thời gian hẹp, không mang xu hướng dài hạn
  - `country`: Chỉ chứa duy nhất 1 giá trị (USA) nên không có giá trị dự đoán
  - `street`: Chứa hơn 4500 địa chỉ riêng biệt, việc mã hóa sẽ gây bùng nổ số chiều và overfitting
  - `statezip`: Thông tin vị trí đã được đại diện bởi cột `city`
  - `sqrt_above`: Loại bỏ do có mối cộng tuyến cao với cột `sqrt_living`

- **Xử lý dữ liệu không hợp lệ**
  - Loại bỏ các dòng có giá trị bị thiếu/rỗng: `price == 0`, `bedrooms == 0` và `bathrooms == 0`

- **Mã hóa biến phân loại**
  - Cột vị trí `city` được mã hóa bằng kỹ thuật **One-Hot Encoding**

- **Phân chia tập dữ liệu**
  - Chia dữ liệu theo tỷ lệ **80% Train** (3.639 mẫu) và **20% Test** (910 mẫu)

### 2. Các mô hình thực nghiệm

#### A. Linear Regression

- Xây dựng thuật toán hồi quy tuyến tính tối ưu bằng Gradient Descent
- Chuẩn hóa đặc trưng bằng cách áp dụng Z-score Normalization để tránh hiện tượng gradient phân kỳ do chênh lệch bậc độ lớn giữa diện tích nhà và các biến còn lại
- Loss Function: MSE $J(\theta) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2$.

#### B. Lasso Regression - $L_1$ Regularization

- Thêm ràng buộc tổng giá trị tuyệt đối trọng số $\alpha \sum |\theta_j|$ vào hàm mục tiêu.
- Triệt tiêu hoàn toàn các trọng số không quan trọng về đúng bằng $0$ (tự động chọn lọc đặc trưng)

#### C. Ridge Regression - $L_2$ Regularization

- Thêm ràng buộc tổng bình phương trọng số $\alpha \sum \theta_j^2$ vào hàm mục tiêu.
- Co nhỏ các hệ số về gần 0 nhưng không loại bỏ hẳn; giúp giải quyết hiện tượng đa cộng tuyến giữa các biến kích thước và số phòng bằng cách chia sẻ trọng số thay vì loại bỏ ngẫu nhiên.

### 3. Kết quả

- Phương pháp đo lường: $R^2$ = $1 - \frac{{SS_res}{SS_tot}}$
  **Linear Regression**: $R^2$ Score: 69.45%
  **Lasso Regression**: $R^2$ Score: 69.49%, Số đặc trưng giữ lại: 41/54
  **Ridge Regression**: $R^2$ Score: 69.49%, Số đặc trưng giữ lại: 52/54

- Trực quan hóa kết quả (Testing từ Linear Regression):
