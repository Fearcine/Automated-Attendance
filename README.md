# Automated Attendance System (Face Recognition)

A simple & efficient face-recognition attendance system built with **Facenet (DeepFace)** + **MTCNN**.  
Perfect for college mini-projects, demos, or small attendance automation setups.

---

## ✨ Features
- 🎥 Real-time face detection  
- 🧠 Face embeddings using **Facenet**  
- 📁 Attendance stored in **SQLite**  
- 👤 Easy face registration using webcam  
- ⚡ Lightweight, no ML knowledge required  

---

## 🛠 Tech Stack
- Python 3.10  
- OpenCV  
- DeepFace  
- MTCNN  
- NumPy  
- SQLite  

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Fearcine/Automated-Attendance.git
cd Automated-Attendance
```
```bash
python -m venv venv 
```
```bash
venv\Scripts\activate
```
```bash 
pip install -r requirements.txt
```
To register New face 
```
python register.py
```
to recognize new face
```
python recognize.py
```
🗄 Database Structure

attendance.db contains two tables:

people
Column	Description
id	Person ID
name	Person name
embedding	Face embedding vector
attendance
Column	Description
id	Entry ID
person_id	Linked to people.id
name	Person name
timestamp	Time marked automatically
📚 Ideal For

College internal projects

Mini-project submissions

-----Type Shi -----
