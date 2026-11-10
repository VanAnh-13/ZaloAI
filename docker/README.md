# DataMate — giải pháp (Zalo AI Challenge 2026)

## Cách chạy
`bash /code/predict.sh` (không tham số): đọc `/data` (chỉ đọc), ghi `/result/submission.jsonl` và
`/result/logs/logs.jsonl`. Mỗi câu một dòng log: `id, answer, elapsed_ms, thinking, code, output`, kèm gợi ý,
phiếu bầu và đặc trưng độ tin cậy. Log hệ thống (vLLM, workspace, tiến độ) nằm ở `/result/system/`.

## Cách tiếp cận
1. **Workspace (CPU, chạy song song lúc vLLM đang load model)** cho từng shop:
   - đọc mọi file csv/xlsx/sqlite, dò dòng header, gộp sheet/file cùng cấu trúc;
   - đọc số kiểu Việt Nam ("55.000đ", "1tr900", "2 xe 4m3") và ngày (serial Excel, dd/mm/yyyy, "1 thg 6 2025");
   - bỏ dòng tổng/ghi chú nằm trong dữ liệu.
   Không có logic nào phụ thuộc business id.
2. **Tầng ngữ nghĩa**:
   - nhận diện vai trò cột (mã đơn, ngày đặt, mặt hàng, thành tiền, trạng thái, khách…) từ mô tả cột trong README và tên cột;
   - áp luật §8 (điền ô trống theo đơn, bỏ dòng trùng trong cùng đơn);
   - dựng mô hình bán hàng chuẩn hoá `S` với các định nghĩa §1: đơn hợp lệ, bỏ đơn huỷ và đơn trả hàng kể cả trả một phần, kỳ theo ngày đặt, voucher/phí ship không tính doanh thu.
   Mọi lựa chọn được ghi vào "schema card" để model kiểm tra. Mô hình được kiểm chứng tự động bằng đáp án của bộ train.
3. **Agent code-act**: model nhận rulebook của split, schema card, câu hỏi và các gợi ý (kỳ thời gian, kiểu đáp án,
   tên trong dữ liệu, cảnh báo kỳ nằm ngoài dữ liệu), rồi viết Python trên `S` / các bảng gốc.
   Mỗi khối code chạy trong một process riêng có timeout. Đáp án cuối cùng do code gọi `answer(type, value)`, nên model
   không chép tay con số nào. Thư viện `dm` cài sẵn các luật ổn định: kỳ thời gian, làm tròn nửa lên, xếp hạng theo mã khi bằng nhau.
4. **Chọn đáp án**:
   - đáp án được chuẩn hoá đúng định dạng từng type;
   - đáp án đáng ngờ (sai kiểu dự đoán, tên không có trong dữ liệu, kỳ nằm ngoài dữ liệu…) được trả lại cho model kiểm tra một lần;
   - nhiều lượt với các biến thể prompt (greedy) rồi bỏ phiếu;
   - một mô hình logistic nhỏ (fit trên dev, `calibration.json`) quyết định abstain khi xác suất đúng thấp.
5. **Tái lập**: temperature 0, seed cố định, thứ tự xử lý cố định. Mỗi đáp án được ghi và fsync ngay. Hệ thống tự giảm
   số lượt khi dự báo không kịp 6 giờ.

## Model (tổng ≤ 15B tham số)
| Model | Nguồn | Revision | Phát hành | License | Tham số |
|---|---|---|---|---|---|
| Qwen3-14B | https://huggingface.co/Qwen/Qwen3-14B | `<commit hash — điền khi tải weights>` | 2025-04-29 | Apache-2.0 | 14.8B |

Không dùng model nào khác (không embedding, không reranker). Mô hình abstain là hồi quy logistic khoảng 10 hệ số, nằm trong
`calibration.json`. Weights nằm ở `/code/models/Qwen3-14B`.

## Dữ liệu và huấn luyện
Phiên bản này không fine-tune LLM và không sinh dữ liệu (xem `/code/training_code/README.md`).
Hệ số của mô hình abstain (nếu có) được fit trên log chạy bộ train. Không dùng dữ liệu public/private test để huấn luyện.

## Thư mục
- `/code/predict.sh` — entry point
- `/code/src/datamate/` — mã suy luận (`core`, `nlu`, `ingest`, `semantics`, `workspace`, `dm`, `agent`, `pipeline.py`)
- `/code/models/` — weights
- `/code/training_code/` — README và manifest
