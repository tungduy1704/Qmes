# Qmes — Daily Workflow

> Quy trình làm việc hàng ngày. Mục tiêu: mỗi ngày kết thúc với repo sạch,
> commit có nghĩa, và code đã được backup lên GitHub.

---

## 🌅 Mở phiên (~1 phút)

- [ ] Mở VS Code → workspace `Qmes` (File → Open Folder)
- [ ] Mở terminal `` Ctrl+` `` — kiểm tra prompt có `(qmatch_env)` và đường dẫn `...\Qmes`
- [ ] **Xác minh interpreter** (xem mục ⚠ bên dưới):
  ```powershell
  python -c "import sys; print(sys.executable)"
  # Phải ra: ...\qmatch_env\Scripts\python.exe
  ```
- [ ] Hỏi Git tình trạng:
  ```powershell
  git status              # còn gì dở dang từ hôm qua?
  git log --oneline -5    # mình đang ở đâu trong dự án?
  git pull                # đồng bộ (thói quen, kể cả khi làm một mình)
  ```
- [ ] Đọc lại mục **"Việc đang treo"** ở cuối file này → chọn việc đầu tiên của ngày

---

## ⚠ Chạy đúng env — quy tắc sống còn

> Hai lần chạy nhầm Python 3.14 hệ thống đã cho ra: meta-dataset rác toàn số 0,
> pivot nhiễm 19 cột, và 2 dataset "biến mất" (106 vs 108). Đừng có lần thứ ba.

**Thứ tự tin cậy:** đường dẫn tuyệt đối > terminal đã activate + `python` trần > nút Run ▶

### Lệnh chuẩn cho script quan trọng / chạy dài (Oracle, LOOCV)
```powershell
& "C:\Users\Dao Duy Tung\Documents\Python\newbie\qmatch_env\Scripts\python.exe" scripts\ten_script.py
```

### Alias cho gọn (dán vào $PROFILE để sống vĩnh viễn)
```powershell
Set-Alias qpy "C:\Users\Dao Duy Tung\Documents\Python\newbie\qmatch_env\Scripts\python.exe"
qpy scripts\ten_script.py
```

### Khóa interpreter cho workspace — `.vscode/settings.json`
```json
{
    "python.defaultInterpreterPath": "C:\\Users\\Dao Duy Tung\\Documents\\Python\\newbie\\qmatch_env\\Scripts\\python.exe"
}
```

### Dấu hiệu nhận biết đã chạy NHẦM env
- Traceback/log chứa đường dẫn `Python314` hoặc `AppData\Roaming\Python`
- Loader báo số dataset khác 108 (vd `loaded 106, skipped 2`)
- `ModuleNotFoundError` với gói chắc chắn đã cài

### Nếu lỡ chạy nhầm
1. Kill script ngay (`Ctrl+C`)
2. **Xóa output dở dang** — đặc biệt file có logic resume
   (vd `Remove-Item results\pivot_mcc_classification.csv`)
3. Xác minh lại interpreter rồi chạy lại từ đầu

---

## 🔨 Trong lúc làm việc

### Quy tắc commit
**Một commit = một đơn vị việc có nghĩa**, mô tả được bằng một câu.

| Tình huống | Commit? |
|---|---|
| Sửa xong một bug | ✅ |
| Viết xong một file test, pytest xanh | ✅ |
| Chạy xong một bước pipeline, có output mới | ✅ |
| "Lưu tạm cho chắc" mỗi 10 phút | ❌ gộp lại |
| Dồn cả ngày thành một cục "update code" | ❌ tách ra |

### Trình tự mỗi lần commit
```powershell
git status      # NHÌN danh sách trước — bắt file rác lọt vào
git add .
git commit -m "Add kernel registry tests"   # động từ đầu câu, tiếng Anh
```

### Khi chạy script dài (Oracle, LOOCV)
- [ ] Dùng **lệnh đường dẫn tuyệt đối / alias `qpy`** (mục ⚠ ở trên)
- [ ] Dòng log đầu phải xác nhận đúng: `loaded 108 datasets`
- [ ] Chạy thử ~10 dataset → kill → chạy lại đọc CHECKPOINT → đạt mới thả chạy hết
- [ ] Trong lúc chờ → làm việc ở **"Việc chen khi chờ"**
- [ ] Script xong → **đọc CHECKPOINT trước**, đạt mới commit kết quả

### Khi cài package mới
```powershell
python -m pip install <tên-gói>     # env đã activate
python -c "import <gói>; print(<gói>.__file__)"   # phải nằm trong qmatch_env
```
- [ ] Ghi tên gói vào mục **"Dependencies chờ chốt"** bên dưới

### Khi gặp lỗi lạ
1. Đọc dòng **cuối** của traceback trước (loại lỗi + thông điệp)
2. Nhìn đường dẫn trong traceback — **có đúng env/repo không?** (lỗi env thì sửa env, không debug code)
3. Chưa hiểu thì dán **nguyên văn** traceback đi hỏi, đừng diễn giải lại

---

## 🌙 Đóng phiên (~2 phút)

- [ ] `git status` — còn gì chưa commit? → commit nốt hoặc ghi vào "Việc đang treo"
- [ ] `git push` — **bắt buộc, đây là backup off-site của cả công trình**
- [ ] Cập nhật mục **"Việc đang treo"** bên dưới (3 dòng là đủ): hôm nay xong gì, mai bắt đầu từ đâu

---

## 📋 Việc đang treo (tự cập nhật mỗi tối)

> _Ví dụ — thay bằng tình trạng thật:_
> - Oracle classification: pivot nhiễm đã xóa, chờ chạy lại bằng đúng env
> - `tests/test_registry.py` chưa viết
> - Mai: chạy Oracle (10 dataset → checkpoint → thả hết), viết test trong lúc chờ

## 🧰 Việc chen khi chờ script chạy

- [ ] Viết tests (`test_registry.py` → `test_data.py` → `test_recommender.py`)
- [ ] Dịch docstring tiếng Việt → tiếng Anh (mỗi lần một module)
- [ ] Cập nhật README / ghi chú thiết kế

## 📦 Dependencies chờ chốt vào `pyproject.toml`

- numpy, pandas, scikit-learn, problexity, ucimlrepo, pytest
- _(gặp gói mới thì thêm vào đây, cuối đợt re-run chốt một lần)_

---

## ⚡ Lệnh cứu hộ

| Tình huống | Lệnh |
|---|---|
| Lỡ `git add` nhầm | `git reset` (bỏ stage, không mất file) |
| Vứt thay đổi chưa commit của 1 file | `git restore <file>` |
| Xem mình đã sửa gì | `git diff` |
| Quên env | `& "..\qmatch_env\Scripts\Activate.ps1"` |
| Python nào đang chạy? | `python -c "import sys; print(sys.executable)"` |
| Chạy chắc chắn đúng env | `& "...\qmatch_env\Scripts\python.exe" script.py` |
| Xóa output nhiễm (resume) | `Remove-Item results\<file>.csv` |
