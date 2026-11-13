# DataMate — giải pháp (Zalo AI Challenge 2026)

## Cách chạy
`bash /code/predict.sh` (không tham số): đọc `/data` (chỉ đọc), ghi `/result/submission.jsonl` và
`/result/logs/logs.jsonl`. Mỗi câu một dòng log: `id, answer, elapsed_ms, thinking, code, output`, kèm gợi ý,
phiếu của từng tầng và đặc trưng độ tin cậy. Log hệ thống (vLLM, workspace, tiến độ) nằm ở `/result/system/`.

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
3. **Trả lời theo cascade ba tầng, mỗi tầng bỏ một phiếu** (tầng rẻ chạy trước):
   - *Tầng 1 — template, không gọi model*: câu hỏi "plain" (bỏ từ chỉ kỳ và từ đệm thì không còn gì) được ánh xạ
     bằng biểu thức chính quy thành một truy vấn trên `S`: doanh thu, số đơn, đơn huỷ, tỉ lệ huỷ, giá trị đơn
     trung bình, ngày cao nhất, % thay đổi.
   - *Tầng 2 — planner, một lần gọi ngắn*: model nhận rulebook của split, các chiều của shop và gợi ý, rồi điền
     một QUERY_SPEC (JSON: metric, kỳ, bộ lọc, xếp hạng) hoặc chuyển câu hỏi cho tầng 3. Một engine tất định thực
     hiện spec (đơn hợp lệ, kỳ theo ngày đặt, làm tròn nửa lên, xếp hạng theo mã khi bằng nhau). Spec có tên không
     có trong dữ liệu, kiểu đáp án lệch câu hỏi hoặc tie ở biên top-k thì bị loại.
   - *Tầng 3 — agent code-act*: model viết Python trên `S` / các bảng gốc trong một phiên sandbox (một tiến trình
     riêng cho mỗi lượt giải, giữ biến giữa các bước, có timeout và giới hạn tài nguyên; code chỉ đọc được file của
     shop đang hỏi, không ghi ra ngoài thư mục tạm, không mạng, không tiến trình con). Đáp án cuối do code gọi
     `answer(type, value)`, nên model không chép tay con số nào. Đáp án đáng ngờ được trả lại cho model kiểm tra
     một lần.
   Không có logic nào phụ thuộc business id hay question id.
4. **Chọn đáp án**:
   - đáp án được chuẩn hoá đúng định dạng từng type;
   - dừng ngay khi hai phiếu trùng nhau; tầng 3 chạy thêm lượt (biến thể prompt, greedy) chỉ khi chưa đủ đồng thuận;
   - không có hai phiếu trùng → abstain. Nếu có `calibration.json`, một mô hình logistic nhỏ trên các đặc trưng
     đồng thuận quyết định thay cho luật này.
5. **Tái lập**: temperature 0, seed cố định, thứ tự xử lý cố định. Mỗi đáp án được ghi và fsync ngay. Khi dự báo
   không kịp 6 giờ, các câu còn lại lấy phiếu đầu tiên thay vì chờ đồng thuận.

Log mỗi câu (`/result/logs/logs.jsonl`): `thinking` là list suy nghĩ nguyên văn của mọi lần gọi model theo thứ tự
(lần gọi planner không bật thinking nên là `null`; phần lập luận của nó nằm ở trường `analysis` của spec trong
`stages`); `code` / `output` là các khối code đã chạy trong sandbox; `stages`, `votes`, `decided_by` cho biết tầng
nào đã bỏ phiếu gì.

## Model (tổng ≤ 15B tham số)
| Model | Nguồn | Revision | Phát hành | License | Tham số |
|---|---|---|---|---|---|
| Qwen3-14B | https://huggingface.co/Qwen/Qwen3-14B | `<commit hash — điền khi tải weights>` | 2025-04-29 | Apache-2.0 | 14.8B |

Không dùng model nào khác (không embedding, không reranker). Mô hình abstain là hồi quy logistic khoảng 10 hệ số, nằm trong
`calibration.json`. Weights nằm ở `/code/models/Qwen3-14B`.

## Dữ liệu và huấn luyện
Phiên bản này không fine-tune LLM (xem `/code/training_code/README.md`).
Nếu có `examples.jsonl` (ví dụ mẫu cho tầng 3) và `calibration.json` (mô hình abstain): cả hai được tạo từ log chạy
bộ **train** và khai báo trong `data_manifest.json`. Không dùng dữ liệu public/private test.

## Thư mục
- `/code/predict.sh` — entry point
- `/code/src/datamate/` — mã suy luận (`core`, `nlu`, `ingest`, `semantics`, `workspace`, `dm`, `agent`, `pipeline.py`)
- `/code/models/` — weights
- `/code/training_code/` — README và manifest
