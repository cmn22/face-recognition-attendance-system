# **Face Recognition Attendance System**
A real-time **face recognition-based attendance system** using **Python, Streamlit, Redis, and InsightFace**. This system detects faces, matches them with stored embeddings, and logs attendance in Redis.

## **🚀 Features**
- **Face Registration**: Users can register their faces and roles (e.g., Student, Teacher).
- **Real-time Face Recognition**: Detects faces and matches them with stored embeddings.
- **Attendance Logging**: Logs check-in and check-out times.
- **Reporting Dashboard**: View attendance logs and generate reports.

---

## **🛠 Project Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/face-recognition-attendance-system.git
cd face-recognition-attendance-system
```

### **2️⃣ Setting up Virtual Environment and Installing Dependencies**
For consistent and reproducible results, it's recommended to use a virtual environment.

#### **Mac/Linux**
1. Open the terminal.
2. Navigate to the project directory:
   ```bash
   cd /path/to/project
3. Create a virtual environment:
   ```bash
   python3 -m venv venv
4. Activate the virtual environment:
   ```bash
   source venv/bin/activate
5. Install required dependencies:
   ```bash
   pip install -r requirements.txt

#### **Windows**
1. Open the Command Prompt or PowerShell. 
2. Navigate to the project directory:
   ```bash
   cd \path\to\project
3. Create a virtual environment:
   ```bash
   python -m venv venv
4. Activate the virtual environment:
   ```bash
   venv\Scripts\activate
5. Install required dependencies:
   ```bash
   pip install -r requirements.txt


### **3️⃣ Configure Environment Variables**
- Create a `.env` file using the example template:
```bash
cp .env.example .env
```
- Open the `.env` file and set your **Redis credentials**:
```
REDIS_HOST=your-redis-host
REDIS_PORT=your-redis-port
REDIS_PASSWORD=your-redis-password
```

### **4️⃣ Run the Application**
```bash
streamlit run Home.py
```
This will start the Streamlit application. Access it at:
```
http://localhost:8501
```

---

## **📂 Project Structure**
```
📦 face-recognition-attendance-system
 ┣ 📂 insightface_model/       # Pretrained InsightFace models
 ┣ 📂 pages/                   # Streamlit pages (Real-time Prediction, Reports)
 ┣ 📜 face_rec.py              # Core face recognition logic
 ┣ 📜 Home.py                  # Main Streamlit dashboard
 ┣ 📜 Report.py                # Attendance reporting module
 ┣ 📜 requirements.txt         # Dependencies
 ┣ 📜 .env                     # Redis credentials (not in Git)
 ┣ 📜 .env.example             # Example env file (in Git)
 ┗ 📜 README.md                # Project documentation
```

---

## **📌 How the System Works**
### **1️⃣ Face Registration**
- Users **register their face** using the camera.
- The system **extracts facial embeddings** using **InsightFace**.
- The extracted embeddings are **stored in Redis** under the key `academy:register`.

### **2️⃣ Real-Time Face Recognition**
- The camera captures faces and extracts **facial embeddings**.
- The system compares these embeddings with **stored embeddings** in Redis.
- If a match is found, **attendance is logged** in Redis under `attendance:logs`.

### **3️⃣ Attendance Logging**
- When a person is detected, an entry is logged as:
  ```
  Name@Role@Timestamp
  ```
- Redis stores this in a **list** under `attendance:logs`.

### **4️⃣ Attendance Reporting**
- Users can view **attendance logs** in the **Reports** tab.
- The system generates a **detailed attendance report**, including:
  - **Check-in time**
  - **Check-out time**
  - **Duration spent**

---

## **📊 Dashboard Overview**
### **🖥 Home Page**
- Start real-time face recognition.
- View detected faces and their roles.

### **📋 Reports Page**
- View registered users.
- Refresh and check attendance logs.
- Generate attendance reports.

---

## **🔗 Future Enhancements**
- Add a **database backup mechanism**.
- Integrate with **cloud storage** for model updates.
- Improve **face recognition accuracy** with better embeddings.

---

## **🙌 Contributors**
- **Chaitanya Malani** (Lead Developer)

For any issues, please raise a GitHub issue or contact me.
