# DataMate Agent — Kế hoạch triển khai (Zalo AI Challenge 2026)

> **Cập nhật 13/11/2026:** luồng trả lời đã được thiết kế lại thành cascade ba tầng bỏ phiếu (template → planner →
> code agent) với sandbox theo phiên; mô tả hiện hành ở [README.md](README.md). Các mục 3.4–3.6 dưới đây là kế hoạch
> ban đầu, giữ lại để đối chiếu.

> Lập ngày 07/11/2026. **Hạn nộp Docker: 16/11/2026 23:59 (GMT+7)**, tức còn khoảng 9 ngày.
> Nguồn: trang đề (Description / Evaluation / Rule / Submission Guidance), Docker guideline PDF, và kết quả phân tích dữ liệu trong `Downloads/train`, `Downloads/public_test`.
> Tab **Data** trên portal cần đăng nhập nên tôi chưa đọc được. Bạn nên đọc lại tab này xem có chi tiết nào không khớp với kế hoạch dưới đây.

---

## 0. Đề bài: những điểm quyết định thiết kế

| Hạng mục | Chi tiết | Hệ quả thiết kế |
|---|---|---|
| Input | `/data/data/<biz>/` (csv / xlsx / sqlite + `README.md`), `rulebook.md` (**riêng cho từng split**), `calendar.csv` (lịch âm–dương), `questions.jsonl` (`id, business_id, question, as_of_date`) | Đọc rulebook + README lúc chạy, không hard-code luật của train |
| Output | `/result/submission.jsonl`: `{"id","type","answer"}`. Các type: `number, percent, string, date, boolean, list, table, unanswerable, abstain` | Bộ formatter phải chặt theo bảng format trong rulebook §9 |
| Điểm | **PA = (số câu đúng − số câu sai) / N**. Abstain hoặc thiếu dòng = 0 điểm | Chỉ trả lời khi P(đúng) > 0.5 → cần hiệu chỉnh độ tin cậy (calibration) |
| So khớp | number/percent khớp tuyệt đối (percent làm tròn 2 chữ số). String được chuẩn hoá (NFC, lowercase, bỏ dấu câu, **giữ dấu tiếng Việt**) và chấp nhận mọi cách viết có trong dữ liệu hoặc mã. List xét cả thứ tự. Table so dòng như một tập, so cột theo tên rồi đến vị trí | Mọi con số phải ra từ code, LLM không được tự tính hay chép lại số |
| Xếp hạng | Giảm dần theo tiêu chí; bằng nhau thì xếp theo **mã tăng dần**. Câu có tie ở biên top-k đã bị loại khỏi test | Nếu tính ra tie ở biên thì gần như chắc là tính sai → hạ độ tin cậy |
| Model | **Tổng tham số mọi model ≤ 15B** (gồm cả embedding, reranker, classifier; MoE tính đủ toàn bộ). Weights public trước 26/10/2026, license cho phép. Được fine-tune và quantize | Chỉ dùng **một model ~14B**. Link thực thể bằng thuật toán (0 tham số) |
| Máy chấm | 2×RTX 5090 (32 GB mỗi card), RAM 512 GB, CUDA ≥ 12.8, **không có mạng**. Base image gợi ý: `vllm/vllm-openai:v0.10.1.1` | Serve vLLM local, không pip/download lúc chạy |
| Thời gian | **6 giờ** cho toàn bộ private test, tính cả lúc load model (Qwen3-14B load mất khoảng 2 phút). Private khoảng **500 câu** (BTC hướng dẫn ước tính "× 500") | Trung bình khoảng 40 giây/câu nếu chạy tuần tự → chạy song song thì đủ cho 3–5 lượt giải mỗi câu |
| Log | `/result/logs/`: mỗi câu một record gồm `id, answer, elapsed_ms, thinking` (nguyên văn, không tóm tắt; gọi model nhiều lần thì là list theo thứ tự), cộng thêm `code, output` nếu có chạy code | Thiết kế logging ngay từ đầu |
| Tái lập | \|PA(bản upload) − PA(BTC chạy lại)\| ≤ 0.03, nếu không giải thích được thì bị loại | Greedy decoding, seed cố định, bỏ phiếu để kết quả ổn định |
| Bị cấm | Gọi API ngoài, người can thiệp, hard-code đáp án, **logic theo business_id hay question_id**, train trên test | Mọi hành vi riêng cho từng shop phải suy ra từ README/file lúc chạy |
| Nộp bài | Public: 5 lượt/ngày. Solution (Docker): tối đa 2 lần, **chỉ chấm lần cuối**. Private test mở 20/11 09:00 và đóng 23/11 23:59 | Nộp một bản an toàn sớm, sau đó nộp bản cuối |

### Dữ liệu (đã phân tích)
- **Train**: 30 shop, 3000 câu có đáp án (`answers.jsonl` có thêm `alias`, là các cách viết hoặc mã được chấp nhận). **Public test**: 14 shop, 1000 câu.
- **Phân bố type (train)**: number 62.8%, percent 19.3%, list 6.4%, date 4.8%, string 2.9%, unanswerable 2.3%, boolean 0.8%, table 0.6%.
- **Phân bố ý định** (ước lượng theo keyword, train / public):

  | Ý định | Train | Public |
  |---|---|---|
  | Doanh thu | 44% | 38% |
  | Xếp hạng, top-k | 13% | 14% |
  | So sánh, % thay đổi | 9% | 10% |
  | Huỷ đơn và số đơn | 14% | 10% |
  | Công nợ | 5% | 8% |
  | Thống kê khách hàng | 3% | 7% |
  | Còn lại: tồn kho, nhập hàng, lãi, lô và hạn dùng, giao hàng | ~12% | ~13% |

  → **Khoảng 70% số câu đi qua chuỗi "dòng bán → đơn hợp lệ → tổng hợp"**. Làm phần này cho thật đúng trước tiên.
- **Câu hỏi bẩn**:
  - Khoảng 26% không có dấu.
  - Teencode: "bnhiu", "dthu", "j z ha", "cko", "mik", "wan".
  - Gõ telex lỗi: "ddieenj", "nguwowif", "cuar".
  - Emoji và phương ngữ ("tháng ni").
  - Viết tắt kỳ: "th7/2025", "T3/2026", "8/26", "2/25", "Q1/2026", "1-2026".
- **File bẩn có chủ đích**:
  - Có dòng filter hoặc tiêu đề phía trên header, có header 2 tầng.
  - Sheet chia theo tháng với nhiều cách đặt tên (`T01_2025`, `04.2025`, `Thang 02 2025`). File tách thành `_1`/`_2`/`_2025`.
  - Ngày lẫn nhiều dạng: serial Excel, `01/02/2025`, `1 thg 6 2025`.
  - Tiền lẫn nhiều dạng: `55.000đ`, `49k`, `38,000`. Số lượng ghi kèm đơn vị (`5 kg`).
  - Khách lẻ ghi là `lẻ`, `kl` hoặc `khách lẻ`. Cách ghi trạng thái đổi theo thời gian (`đã lấy`, `Thành công`, `Đã hoàn thành`).
  - Dòng trả hàng ghi số âm, có cột trống để bẫy, có shop dùng SQLite của máy POS.
- **README** mỗi shop dùng bộ từ vựng mô tả cột khá chuẩn: 127 mô tả cho hàng trăm tên cột khác nhau ("mã đơn", "tên hàng", "thành tiền sau giảm", "trạng thái đơn", ...). Mục **Quy ước** ghi ngày hết dữ liệu, quy đổi đơn vị, cách ghi nợ, trả hàng, tồn kho...
- **Rulebook khác nhau giữa các split**. Public có thêm ngành Mẹ & bé, Sửa xe, Cửa hàng điện thoại và nhiều định nghĩa mới (khách trung thành, khách sộp, giờ cao điểm, nhà cung cấp chính...). Ngược lại, train có vài luật mà public không có. → **Private gần như chắc chắn có ngành và luật mới**, nên agent phải tổng quát.
- Các shop phức tạp có nhiều sổ nối với nhau qua STT, IMEI, tên hoặc số điện thoại (ví dụ sổ góp phải tự tính lịch kỳ góp). Một schema cố định không phủ hết được, nên **bắt buộc có code-agent tổng quát**.

---

## 1. Chiến lược: lớp dữ liệu tất định + LLM lập kế hoạch và viết code + bỏ qua có hiệu chỉnh

1. **LLM không bao giờ tự tính hay chép số.** Đáp án cuối do *code* gọi `answer(type, value)`, nên giá trị đi thẳng từ kết quả tính vào submission.
2. **Nạp và làm sạch mỗi shop một lần** (ingest) thành bảng sạch kèm một "schema card". Agent chỉ làm việc trên bảng sạch.
3. **Luật ổn định thì viết thành helper có test**: kỳ thời gian, đơn hợp lệ, doanh thu, quy đổi đơn vị, xếp hạng và tie-break, làm tròn. **Luật mới hoặc lạ thì để LLM đọc rulebook nguyên văn và tự viết code.**
4. **Giải nhiều lượt độc lập, chạy qua bộ kiểm tra để ra độ tin cậy, dưới ngưỡng thì abstain.** Với thang +1/0/−1, một câu abstain đúng lúc có giá trị như tránh được −1.
5. **Không có logic theo ID.** Mọi thứ suy ra từ README, file và rulebook của split lúc chạy.

---

## 2. Kiến trúc

```
/code/predict.sh
  ├─ khởi động vLLM (Qwen3-14B) ở nền → chờ /health
  └─ python -m datamate.run --data /data --out /result
        │
        ├─ [A] INGEST (CPU, song song theo shop, ~1–3 phút)
        │     README parser → loaders → chuẩn hoá cột/giá trị → bảng canonical + bảng thô đã sạch
        │     → profile (khoảng ngày, giá trị phân loại, từ điển thực thể) → schema card (markdown)
        │     → /tmp/ws/<biz>/*.parquet
        │
        ├─ [B] AGENT cho từng câu (async, 32–64 câu đồng thời, nhóm theo shop để tận dụng prefix cache)
        │     1. Tiền xử lý: chuẩn hoá câu, parse kỳ → ngày cụ thể, link thực thể (fuzzy), đoán type
        │     2. Vòng code-act: LLM viết Python dùng helper `dm` → sandbox (fork, timeout) → quan sát → … → answer()
        │     3. Chuẩn hoá và kiểm tra đáp án (format, làm tròn, tên có trong dữ liệu, tie ở biên top-k…)
        │     4. K lượt biến thể → bỏ phiếu → độ tin cậy → trả lời / unanswerable / abstain
        │     5. Ghi ngay submission.jsonl + log (flush + fsync)
        │
        └─ [C] TIME MANAGER: deadline tổng ~5h30; trễ thì giảm K; câu chưa xong → abstain
```

---

## 3. Thành phần chi tiết

### 3.1 Ingest (quan trọng nhất: sai ở đây thì mọi câu đều sai)
Loader viết tổng quát, **không viết riêng cho shop nào**.
- **Đọc file**:
  - xlsx: đọc mọi sheet, dùng calamine cho nhanh, chuyển sang openpyxl khi cần merged cells.
  - csv: tự dò BOM, encoding, delimiter.
  - sqlite: mở bằng `file:...?mode=ro&immutable=1` vì `/data` là read-only.
- **Dò header**: chấm điểm 10 dòng đầu theo mức khớp với tên cột trong README. Xử lý được dòng filter/tiêu đề phía trên, dòng trống, header 2 tầng và merged cells.
- **Gộp sheet/file cùng loại** (cùng tập cột) thành một bảng. Lưu lại tháng/năm lấy từ tên sheet hoặc tên file để dùng cho luật "ngày không ghi năm".
- **Parse giá trị**:
  - Ngày: serial Excel (gốc 1899-12-30), `dd/mm/yyyy`, `1 thg 6 2025`, chuỗi datetime, có hoặc không có năm.
  - Số kiểu VN: `55.000đ`, `49k`, `38,000`, `1.234,5`, số âm. Chuỗi `5 kg` tách thành số 5 và đơn vị.
- **Luật §8 của rulebook**:
  - Forward-fill ô trống trong nhóm dòng của cùng một đơn.
  - Bỏ dòng trùng y hệt **trong cùng một đơn**. Bảng không ghi theo đơn thì giữ cả dòng trùng.
  - Sổ không có cột trạng thái thì đọc trạng thái từ cột Ghi chú.
  - Ngày không ghi năm thì lấy năm của sheet.
- **Gộp trạng thái về cấp đơn**: đơn có *bất kỳ* dòng trả hàng nào (kể cả nằm ở sheet hay tháng khác, kể cả phiếu trả ghi sau as_of) thì cả đơn là "trả hàng". Tương tự cho "huỷ".
- **Map cột sang khái niệm canonical**:
  - Dựa vào mô tả cột trong README, tra từ điển "mô tả → khái niệm" xây **chỉ từ README của train**.
  - Mô tả lạ thì gọi LLM **một lần cho mỗi shop**.
  - Kiểm tra chéo bằng dữ liệu, ví dụ `line_total ≈ qty × unit_price − discount`.
- **Chuẩn hoá giá trị phân loại** (từ điển + một lần gọi LLM cho mỗi shop):
  - Trạng thái → {hoàn thành, huỷ, trả hàng, đang giao}.
  - Nhận diện khách lẻ.
  - Hình thức thanh toán: tiền mặt, chuyển khoản, ví, COD, ghi nợ.
  - Kênh bán.
- **Quy đổi đơn vị**: đọc từ README ("1 thùng = 24 lon") hoặc bảng quy đổi để tính `base_qty`.
- **Đầu ra**:
  - Bảng canonical khi nhận diện được: `sales_lines, orders, products, customers, payments, purchases, stock, expenses, deliveries, price_history, batches`.
  - **Luôn giữ thêm mọi bảng thô đã làm sạch** (cả tên cột gốc lẫn tên chuẩn) để xử lý các sổ đặc thù như sổ góp, sổ bảo hành, sổ sửa xe.
- **Schema card** (~1.5–3k token cho mỗi shop):
  - Mỗi bảng: nguồn, số dòng, khoảng ngày, danh sách cột (tên gốc → ý nghĩa → kiểu, ví dụ giá trị, các giá trị phân loại).
  - Ngày hết dữ liệu và mục "Quy ước" của README (nguyên văn).
  - **Danh sách thông tin KHÔNG có** trong dữ liệu (để nhận diện câu unanswerable).
- **Kiểm thử bằng đáp án train (oracle)**:
  - Viết hàm tính cho các mẫu câu phổ biến nhất (doanh thu tháng, số đơn, tỉ lệ huỷ, top-k bán chạy, nợ đến hết ngày X...) bằng helper.
  - So kết quả với `answers.jsonl` trên cả 30 shop. Chỗ nào lệch là bug ở loader hoặc helper.
  - Đây là công cụ dev, không chạy lúc nộp bài.

### 3.2 Thư viện helper `dm` (tài liệu API đưa vào prompt)
Thứ tự ưu tiên theo tần suất câu hỏi:
- **Thời gian**: `dm.period(text|spec, as_of)`, `dm.prev_period(...)`, `dm.same_period_last_year(...)`, `dm.tet_range(year)` (tra `calendar.csv`), `dm.shift_of(time)`. Các hàm này cài đúng §2 của rulebook:
  - Tuần bắt đầu từ thứ Hai. "Tuần rồi" là trọn tuần trước.
  - "Tháng này" tính từ ngày 1 đến hôm nay. So với tháng trước thì lấy cùng số ngày.
  - "Quá/trên N" không gồm N. "N ngày gần nhất" tính cả hôm nay.
  - Tháng hoặc quý không ghi năm thì lấy năm nay, trừ khi rơi sau hôm nay.
- **Đơn và doanh thu**: `dm.lines(...)`, `dm.valid_orders(start, end, as_of)`, `dm.revenue(...)`, `dm.order_count(...)`, `dm.cancel_rate(...)`.
  - Đơn hợp lệ: bỏ đơn huỷ và đơn trả hàng (kể cả trả một phần), xác định kỳ theo ngày đặt.
  - Doanh thu là tổng thành tiền dòng: không trừ giảm giá cả đơn, không cộng VAT hay phí giao.
  - Tỉ lệ huỷ có mẫu số là mọi đơn đặt trong kỳ.
- **Thực thể**:
  - `dm.find(kind, text)` trả về các ứng viên kèm điểm (so sánh không dấu, token-set, theo mã).
  - `dm.customer_key()` lấy theo mã nếu shop có mã, không thì theo tên đã chuẩn hoá; khách lẻ không được tính.
- **Xếp hạng**: `dm.rank(df, by, k)` xếp giảm dần, tie theo mã tăng dần. Trả kèm **cảnh báo khi có tie ở biên k**.
- **Số học**:
  - `dm.round_money()` dùng ROUND_HALF_UP. **Không dùng `round()` của Python** vì hàm đó làm tròn kiểu banker's.
  - `dm.pct(a, b)` và `dm.pct_change(new, old)` làm tròn 2 chữ số.
- **Công nợ, kho, nhập hàng** (bản cho các dạng dữ liệu phổ biến): `dm.debt_as_of`, `dm.debt_by_customer`, `dm.stock_as_of`, `dm.purchase_total`, `dm.weighted_avg_purchase_price`...
- **`answer(type, value, reason=None)`**: điểm kết thúc duy nhất của mỗi lượt giải.
- Mọi helper có docstring trích luật tương ứng. Prompt dặn rõ: *nếu rulebook trong ngữ cảnh nói khác helper thì theo rulebook và tự viết code.*

### 3.3 Tiền xử lý câu hỏi (tất định, đưa vào prompt dưới dạng gợi ý)
- **Chuẩn hoá**: NFC, từ điển teencode/viết tắt (xây từ train), giải mã telex lỗi ("ddieenj" → "điện"). Luôn giữ cả bản gốc.
- **Parser kỳ thời gian tiếng Việt**: đổi cách nói tương đối thành khoảng ngày tuyệt đối theo `as_of_date`.
  - Ví dụ: "tuần rồi" với as_of 2026-03-04 thành 2026-02-23 → 2026-03-01.
  - Trường hợp mơ hồ ("8/26", "2/25") thì đưa ra nhiều ứng viên, xếp theo thứ tự ưu tiên.
- **Link thực thể**: top-3 ứng viên kèm điểm cho từng loại: sản phẩm, nhóm hàng, khách, nhân viên, nhà cung cấp, kênh bán, hình thức thanh toán.
- **Đoán type kỳ vọng**:

  | Dấu hiệu trong câu hỏi | Type |
  |---|---|
  | "bao nhiêu %" | percent |
  | "ngày nào" | date |
  | "tháng nào" | date dạng `YYYY-MM` |
  | "có … không" | boolean |
  | "top k", "những … nào" | list |
  | "mỗi người … bao nhiêu" | table |
  | Hỏi nhiều số | list, theo đúng thứ tự câu hỏi nêu |

### 3.4 Vòng lặp agent (code-act)
- **Thứ tự prompt** (sắp xếp để tối ưu prefix cache):
  1. System: vai trò, giao thức, bảng format đáp án.
  2. Rulebook của split, nguyên văn.
  3. Tài liệu API của `dm`.
  4. Hai đến bốn ví dụ mẫu lấy từ train, thể hiện *quy trình* chứ không phải đáp án.
  5. Schema card của shop.
  6. Câu hỏi gốc, bản đã chuẩn hoá, `as_of_date`, cùng các gợi ý về kỳ, thực thể và type.
- **Mỗi lượt**: model suy nghĩ rồi trả **đúng một khối ```python```**. Sandbox chạy khối đó và trả về stdout hoặc lỗi (cắt còn khoảng 2k ký tự). Tối đa khoảng 6 lượt. Lượt giải kết thúc khi code gọi `answer(...)`.
- **Sandbox**:
  - Mỗi lần chạy code là một process con fork từ worker đã nạp sẵn dữ liệu của shop. Nhờ copy-on-write nên nhanh mà vẫn cách ly.
  - Timeout khoảng 20–30 giây. `/data` chỉ đọc, file tạm ghi vào `/tmp`.
  - Lưu ý: Windows không có `fork`, nên dev phần này trên Linux (WSL2 hoặc máy GPU thuê).
- **Logging**: lưu lại toàn bộ thinking, code và output để ghi log theo yêu cầu.

### 3.5 Chuẩn hoá và kiểm tra đáp án
- **Format theo type**:
  - Tiền làm tròn về số nguyên kiểu ROUND_HALF_UP.
  - Percent và các số trung bình làm tròn 2 chữ số.
  - Date theo ISO.
  - List và table đúng cấu trúc. Table có cột đầu là tên đối tượng, các cột sau là số đo theo thứ tự câu hỏi nêu.
- **String**: phải là một tên hoặc mã có trong dữ liệu của shop. Nếu không, map về tên gần nhất hoặc hạ độ tin cậy.
- **Cờ nghi ngờ**:
  - Kết quả NaN hoặc rỗng.
  - Doanh thu bằng 0 trong một kỳ có dữ liệu.
  - Kỳ hỏi nằm ngoài khoảng dữ liệu.
  - Có tie ở biên top-k.
  - Type lệch với type kỳ vọng.
  - Code phải sửa lỗi nhiều lần.

### 3.6 Bỏ phiếu, độ tin cậy, abstain, unanswerable
- **Bỏ phiếu thích ứng**:
  - Chạy trước 2 lượt với 2 biến thể prompt, cả hai đều greedy. Nếu hai lượt trùng đáp án và không có cờ nghi ngờ thì chốt.
  - Nếu lệch nhau thì chạy thêm 2–3 lượt với các biến thể khác: "dùng helper", "tính từ bảng thô", bộ ví dụ khác, bật hoặc tắt thinking.
- **Tạo đa dạng bằng biến thể prompt thay vì temperature**, để kết quả vẫn tái lập được.
- **Độ tin cậy**:
  - Dùng hồi quy logistic nhỏ trên các đặc trưng: tỉ lệ phiếu, số đáp án khác nhau, cờ nghi ngờ, điểm link thực thể, type, số lần sửa lỗi.
  - Fit trên tập dev (giữ riêng theo shop).
  - Chỉ trả lời khi p̂ > τ. Chọn τ để PA trên dev cao nhất (thường khoảng 0.5–0.6), và để thêm biên an toàn cho ngành mới.
- **Unanswerable**:
  - Kiểm tra tất định xem kỳ hỏi, *và cả kỳ so sánh*, có nằm **hoàn toàn** ngoài khoảng dữ liệu không.
  - Kiểm tra loại thông tin không có trong schema card (địa chỉ, tay nghề...).
  - Chỉ trả lời unanswerable khi đa số phiếu đồng ý.
  - Kỳ nằm một phần trong dữ liệu thì tính trên phần có dữ liệu (§7). Trả lời unanswerable sai cũng bị −1.

### 3.7 Runtime
- **vLLM**: benchmark hai phương án trên máy 2×5090 thuê.
  - **A** (giống mẫu của BTC): Qwen3-14B BF16, TP=2.
  - **B**: Qwen3-14B-FP8, mỗi GPU một server (DP=2), client tự chia tải. Có thể nhanh hơn A vì cặp GPU của BTC không có P2P. Cần kiểm tra FP8 chạy được trên sm_120 với vLLM 0.10.1.1.
- **Tham số vLLM**: `--max-model-len 32768`, bật prefix caching, `--seed` cố định, temperature 0, `--reasoning-parser qwen3` để lấy `reasoning_content` ghi log. Cân nhắc thêm `--kv-cache-dtype fp8`.
- **Ràng buộc KV cache**: mẫu của BTC chỉ có khoảng 134k token KV cache (TP=2, BF16, util 0.8). Vì vậy prompt phải gọn và nên nhóm câu theo shop.
- **Ngân sách token** (ước lượng thô, cần đo thật): 500 câu × K≈3 lượt × ~3 vòng × ~1–2k token output, tức khoảng 4–9M token. Ở tốc độ ~1000 token/s thì mất 1.5–3 giờ, vẫn trong giới hạn 6 giờ.
- **Time manager**:
  - Đo tốc độ sau khoảng 20 câu đầu.
  - Nếu dự báo vượt 5h30 thì giảm K và tắt thinking ở các lượt phụ.
  - Hết giờ thì abstain các câu còn lại.
- **Không bao giờ làm chết cả run**: bọc try/except ở mọi tầng; câu nào lỗi thì abstain câu đó. Ghi từng câu ngay khi xong.

---

## 4. Chọn model
- **Chính: Qwen3-14B** (14.8B, Apache-2.0, phát hành 4/2025).
  - BTC đã kiểm chứng trên đúng stack (vLLM 0.10.1.1, load khoảng 2 phút).
  - Mạnh về code và tool-use, tiếng Việt khá, có thinking mode.
  - Model này dùng gần hết ngân sách 15B, nên **không dùng thêm embedding hay reranker**. Link thực thể bằng rapidfuzz/unidecode.
- **Phương án A/B: Qwen3.5-9B** (phát hành 3/2026, benchmark công bố tốt về tool-use và coding).
  - Kiến trúc mới cần vLLM mới hơn 0.10.1.1, và đã có báo cáo trục trặc trên 5090.
  - Chỉ thử khi pipeline đã ổn định, và phải tự dựng base image.
- **Tránh** các model MoE lớn (30B-A3B...). Luật tính đủ toàn bộ tham số nên các model này vượt 15B.
- **Thí nghiệm bắt buộc**: so thinking bật và tắt trên bộ hồi quy 300 câu, xét cả PA lẫn thời gian.
- Kiểm tra thêm xem có model ≤15B nào mới hơn ra trước 26/10/2026 không. Kiến thức của tôi chỉ đến khoảng giữa 2026.
- README phải khai báo: tên model, nguồn HF, revision cố định, ngày phát hành, license, số tham số.

---

## 5. Đánh giá và vòng lặp phát triển
- **Scorer cục bộ** bám đúng luật chấm: alias, chuẩn hoá chuỗi, list có thứ tự, table so tập dòng và so cột theo tên rồi đến vị trí.
- **Chia tập dev**:
  - Train 30 shop: 22 shop làm "dev" (dùng để viết ví dụ mẫu và từ điển), 8 shop làm "holdout" (chỉ dùng để đo).
  - Thêm phép thử "ngành chưa thấy": bỏ hẳn một ngành (ví dụ nhà thuốc) khỏi ví dụ mẫu và từ điển, rồi đo riêng trên ngành đó.
- **Bộ hồi quy nhanh**: khoảng 300 câu phân tầng theo ngành × ý định × type. Mỗi lần sửa chạy khoảng 10–15 phút.
- **Báo cáo**:
  - PA, độ phủ, accuracy khi trả lời; chia theo type, ý định, shop.
  - Đường cong PA theo ngưỡng τ.
  - Các lỗi hàng đầu kèm trace.
- **Public leaderboard** (5 lượt/ngày):
  - Dùng để so sánh cấu hình và ngưỡng τ ở mức tổng thể.
  - Một bản nộp "trả lời hết, không abstain" cho biết accuracy = (PA + 1) / 2.
  - Khi nộp bằng pipeline ≤15B thì chọn model type **SLM (≤15B)**.
  - Không dò từng câu (Rule 6).
- **Public test chỉ dùng để đo**: không lấy làm ví dụ mẫu, nguồn từ điển hay dữ liệu train.

---

## 6. Lộ trình (07/11 → 16/11)

**Nguyên tắc: có pipeline end-to-end chạy được trong Docker sớm nhất có thể, sau đó mới tối ưu độ chính xác.**

| Ngày | Việc | Xong khi |
|---|---|---|
| **T7 07/11 – CN 08/11** | Dựng khung repo, scorer, loader v0, agent "ngây thơ" (prompt + pandas, chưa có helper), ghi submission và log. Thuê 1 GPU (5090/4090) chạy vLLM Qwen3-14B làm endpoint dev. Viết Dockerfile v0 | Có PA baseline trên bộ 300 câu; nộp public lần đầu |
| **T2 09/11 – T4 11/11** | Loader đầy đủ, README parser, chuẩn hoá phân loại. Bộ helper `dm` (period, valid_orders, revenue, rank, rounding, find). Oracle kiểm thử bằng answers train. Agent code-act có gợi ý tiền xử lý và ví dụ mẫu | Oracle khớp ≥ 95% các mẫu câu phổ biến; PA dev tăng rõ so với baseline |
| **T5 12/11 – T6 13/11** | Bỏ phiếu, bộ kiểm tra, hiệu chỉnh τ, xử lý unanswerable. Xử lý ngành mới (sổ đặc thù). Thí nghiệm thinking bật/tắt; benchmark TP=2 và DP=2. **Chạy Docker trên 2×5090 thuê với `--network none`** | Docker chạy hết public test; đo được thời gian; so sánh 2 lần chạy |
| **T7 14/11** | Đóng băng v1. **Nộp solution lần 1 làm lưới an toàn** (chỉ lần nộp cuối được chấm) | Tải lại link Drive từ máy khác và kiểm md5 OK |
| **CN 15/11** | Cải tiến cuối (prompt, τ, từ điển). Viết README, training_code, data_manifest. Test tái lập | Bộ hồi quy không tụt; ΔPA giữa 2 lần chạy ≤ 0.01 |
| **T2 16/11** | Đóng băng, build, kiểm tra cuối, upload (pigz + md5). Nộp lần 2 **trước 18:00** để còn thời gian xử lý sự cố | Đã nộp; tải lại kiểm md5 OK |
| **20/11 09:00 – 23/11** | Chạy **đúng image đã nộp** trên private test (máy 2×5090 thuê), nộp `submission.jsonl` | Nộp sớm, giữ lại toàn bộ log |

Nếu làm theo nhóm, chia 3 mảng: (1) data/loader + oracle; (2) agent, prompt, bỏ phiếu; (3) infra: vLLM, Docker, thời gian chạy, log.

**Môi trường dev**: laptop không có GPU NVIDIA. Đề xuất:
- Thuê **một GPU suốt giai đoạn dev** và code thẳng trên đó qua VS Code Remote-SSH (môi trường Linux, có `fork`, giống máy chấm).
- Chỉ thuê **2×5090** khi cần test Docker cuối và khi chạy private test.
- Nếu tìm được API host Qwen3-14B thì dùng làm phương án dự phòng. Client LLM viết theo chuẩn OpenAI-compatible để đổi backend dễ dàng.

---

## 7. Docker và checklist nộp bài
- Dockerfile dựa trên mẫu của BTC:
  - `FROM vllm/vllm-openai:v0.10.1.1`, đặt `ENTRYPOINT []`.
  - `ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_P2P_DISABLE=1 PYTHONHASHSEED=0`.
  - Cài thêm `nvidia-nccl-cu12==2.27.7` (sửa lỗi treo khi khởi tạo 2 GPU).
  - COPY weights thành một layer riêng để build lại nhanh khi chỉ sửa code.
- Cấu trúc `/code`: `predict.sh`, `README.md`, `requirements.txt`, `src/`, `models/`, `training_code/` (gồm `README.md`, `data_manifest.json` có SHA-256; **ghi rõ nếu không fine-tune**).
- `predict.sh`:
  - Chạy không cần tham số, tự tạo `/result` và `/result/logs`.
  - Mọi tham số (model, seed, temperature, concurrency) cố định trong file.
  - Kill vLLM khi chạy xong.
- Test bằng **đúng lệnh chấm của BTC**:
  ```bash
  docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
    -v /path/to/public_test:/data:ro -v /path/to/output:/result \
    zaic2026_datamate_<team>:final /code/predict.sh
  ```
- Kiểm tra trước khi nộp:
  - [ ] Chạy hết không lỗi khi không có mạng.
  - [ ] `submission.jsonl` có đủ mọi câu.
  - [ ] Mỗi câu có đủ log `id, answer, elapsed_ms, thinking`.
  - [ ] Thời gian load + (trung bình mỗi câu × 500) < 6 giờ, có biên an toàn.
  - [ ] Chạy 2 lần cho PA lệch nhau ≤ 0.03 (mục tiêu ≤ 0.01).
  - [ ] Image linux/amd64; file `.tar` chưa nén ≤ 100 GB.
- Upload:
  - `set -o pipefail; docker save … | pigz > zaic2026_datamate_<team>.tar.gz`, rồi `md5sum` ra file `.md5`.
  - Upload lên Google Drive (rclone từ máy cloud), chia sẻ link public.
  - Tải lại trên máy khác và chạy `md5sum -c`. Gửi kèm kích thước `.tar` chưa nén.
  - Dự trù vài giờ để upload khoảng 35 GB.

---

## 8. Rủi ro và phương án dự phòng

| Rủi ro | Phòng tránh |
|---|---|
| Loader hỏng với định dạng chưa gặp ở private | Fallback: schema card hiển thị preview file thô để agent tự đọc bằng pandas. try/except theo từng file. Oracle kiểm thử trên 30 shop train |
| Rulebook private có luật mới hoặc sửa luật cũ | Luôn đưa rulebook nguyên văn vào prompt. Helper có docstring trích luật. (Tuỳ chọn) lúc chạy, diff rulebook với các bản đã biết và đẩy luật mới lên đầu prompt |
| Vượt 6 giờ | Time manager, ghi từng câu ngay, K thích ứng, đo trên 2×5090 thật |
| Kết quả lệch giữa 2 lần chạy (GPU batching) | Greedy decoding, biến thể prompt thay temperature, bỏ phiếu, thứ tự xử lý cố định, test chạy 2 lần |
| Trả lời sai nhiều gây điểm âm | Hiệu chỉnh τ, cờ nghi ngờ, abstain mặc định khi có lỗi |
| Vi phạm luật (logic theo ID, dùng test để train) | Code review riêng mục này. Từ điển và ví dụ mẫu chỉ lấy từ train |
| Lỗi hạ tầng (NCCL, shm, tokenizer cần mạng) | Dockerfile theo mẫu BTC, test `--network none` từ sớm (13/11) |
| Upload chậm hoặc hỏng | Nộp bản an toàn 14/11, kiểm md5 từ máy khác |

---

## 9. Cấu trúc repo đề xuất
```
ZaloAI/
├── PLAN.md
├── datamate/                # = /code/src trong image
│   ├── run.py               # entry: đọc /data, điều phối, ghi /result
│   ├── ingest/              # readme_parser.py, loaders.py, normalize.py, schema_map.py, profile.py
│   ├── dm/                  # helper: time.py, sales.py, entities.py, rank.py, money.py, debt.py, stock.py
│   ├── agent/               # prompts/, preprocess.py, loop.py, sandbox.py, vote.py, calibrate.py
│   ├── llm.py               # client OpenAI-compatible (vLLM local / API dev)
│   └── logs.py
├── eval/                    # scorer.py, oracle.py, splits.json, report.py
├── docker/                  # Dockerfile, predict.sh, README.md (khai báo model)
├── training_code/           # README, data_manifest.json, script dựng từ điển
└── tests/                   # unit test cho period parser, number parser, scorer, helper
```

---

## 10. Tuỳ chọn nếu còn dư thời gian (sau 12/11 và chỉ khi pipeline đã ổn định)
- **Distillation / LoRA SFT**:
  - Dùng teacher (Claude hoặc GPT qua API; được phép dùng để sinh dữ liệu nhưng phải lưu prompt và response) giải câu train bằng đúng harness.
  - Giữ lại các trajectory cho đáp án đúng, rồi SFT Qwen3-14B.
  - Phải đóng gói đủ code, data, log và checkpoint vào image.
- **Câu hỏi tổng hợp**: tự sinh paraphrase dạng teencode, không dấu, telex lỗi để test độ bền.
- **Diff rulebook lúc chạy**: so rulebook của split với các bản đã biết để làm nổi bật luật mới cho LLM.
