# Portfolio Correlation Heatmap

Mini project สำหรับวิเคราะห์ความสัมพันธ์ระหว่างหุ้น ETF ดัชนี หรือสินทรัพย์หลายประเภท เพื่อศึกษาการกระจายความเสี่ยงของพอร์ต

## ฟีเจอร์

- รับ ticker หลายตัวผ่าน comma หรือขึ้นบรรทัดใหม่
- ดึงข้อมูลราคาด้วย `yfinance`
- รองรับ Simple Return และ Log Return
- คำนวณ Pearson Correlation Matrix จาก Return
- สร้าง Seaborn Heatmap
- เลือกซ่อนครึ่งบนของ Matrix
- แสดงคู่สินทรัพย์ที่มี Correlation สูงหรือต่ำ
- กรองคู่สินทรัพย์ตาม Absolute Correlation
- เปรียบเทียบราคาแบบ Normalize ค่าเริ่มต้นเป็น 100
- ดาวน์โหลด Heatmap, Correlation Matrix, Return และ Price Data
- จัดการ ticker ที่ไม่พบข้อมูลหรือมีข้อมูลไม่เพียงพอ

## โครงสร้างโปรเจกต์

```text
correlation-heatmap-dashboard/
├── app.py
├── learning_notebook.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## วิธีติดตั้ง

แนะนำ Python 3.11 หรือใหม่กว่า

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

## วิธีรัน

```bash
streamlit run app.py
```

เปิด URL ที่ Streamlit แสดงใน Terminal โดยทั่วไปคือ

```text
http://localhost:8501
```

## ตัวอย่าง Ticker

```text
NVDA, AAPL, MSFT, GOOGL, AMZN, META, AVGO, TSM, VOO, GLD
```

ตัวอย่างประเภทอื่น:

| สินทรัพย์ | Ticker |
|---|---|
| NVIDIA | `NVDA` |
| S&P 500 ETF | `VOO` |
| Nasdaq-100 ETF | `QQQ` |
| Gold ETF | `GLD` |
| Bitcoin | `BTC-USD` |
| PTT | `PTT.BK` |
| S&P 500 Index | `^GSPC` |

## แนวคิดหลัก

### 1. คำนวณ Return

ไม่ควรคำนวณ Correlation จากระดับราคาดิบ เพราะราคาหลายสินทรัพย์อาจมีแนวโน้มเพิ่มขึ้นตามเวลาและทำให้ความสัมพันธ์ดูสูงเกินจริง

Simple Return:

```python
returns = prices.pct_change()
```

Log Return:

```python
log_returns = np.log(prices / prices.shift(1))
```

### 2. Correlation Matrix

```python
correlation = returns.corr(method="pearson")
```

ค่าที่ได้อยู่ระหว่าง -1 ถึง 1

- ใกล้ `1`: เคลื่อนไหวไปทางเดียวกันมาก
- ใกล้ `0`: ไม่มีความสัมพันธ์เชิงเส้นชัดเจน
- ใกล้ `-1`: มักเคลื่อนไหวสวนทางกัน

### 3. Seaborn Heatmap

```python
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0,
)
```

## การตีความสำหรับ Diversification

ตัวอย่างเชิงแนวคิด:

- หุ้นเทคโนโลยีหลายตัวอาจมี Correlation สูง เพราะได้รับผลกระทบจากปัจจัยคล้ายกัน
- ETF หุ้นสหรัฐฯ อาจสัมพันธ์กับหุ้นขนาดใหญ่ในดัชนี
- ทองคำ พันธบัตร หรือสินทรัพย์ประเภทอื่นอาจมี Correlation ต่ำกว่าในบางช่วง
- Correlation สามารถเพิ่มขึ้นในช่วงตลาดตกหรือวิกฤตได้

Correlation เพียงอย่างเดียวไม่เพียงพอสำหรับตัดสินคุณภาพพอร์ต ควรดูร่วมกับ:

- Volatility
- Maximum Drawdown
- Position Weight
- Sector Concentration
- Currency Exposure
- Investment Horizon

## ลำดับการเรียนรู้

1. เปิด `learning_notebook.ipynb`
2. ทดลองดึงราคาหุ้น 5 ตัว
3. คำนวณ Return
4. ตรวจ Missing Values
5. สร้าง Correlation Matrix
6. สร้าง Heatmap
7. หา Top correlated pairs
8. รัน `app.py` เพื่อดู Dashboard เต็มรูปแบบ

## ไอเดียต่อยอด

- Rolling Correlation เช่นย้อนหลัง 60 วัน
- Correlation ก่อนและหลังเหตุการณ์สำคัญ
- Hierarchical Clustering และ Clustered Heatmap
- Minimum Variance Portfolio
- Risk–Return Scatter Plot
- Maximum Drawdown Matrix
- Correlation แยก Bull Market และ Bear Market
- เพิ่ม Asset Allocation Simulator

## ไอเดียคอนเทนต์ TikTok

**Hook**

> ถือหุ้นหลายตัว ไม่ได้แปลว่ากระจายความเสี่ยง ถ้าทุกตัวขึ้นลงพร้อมกัน

**โครงเรื่อง**

1. เลือกหุ้น 10 ตัวในพอร์ต
2. ดึงราคาย้อนหลัง
3. เปลี่ยนราคาเป็น Return
4. คำนวณ Correlation
5. แสดง Heatmap
6. ชี้คู่ที่เคลื่อนไหวคล้ายกันมากที่สุด
7. ย้ำว่า Correlation เปลี่ยนได้ตามช่วงเวลา

## คำเตือน

โปรเจกต์นี้จัดทำเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน และข้อมูลย้อนหลังไม่ได้รับประกันผลลัพธ์ในอนาคต
