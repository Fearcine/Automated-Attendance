import pandas as pd
import numpy as np
from datetime import datetime
import cv2


EXCEL_FILE = "attendance.xlsx"

def show_today():
    try:
        df = pd.read_excel(EXCEL_FILE)
    except:
        print("No attendance file.")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    df = df[df["Date"] == today]

    if df.empty:
        print("No attendance today.")
        return

    img = 255 * np.ones((400, 600, 3), dtype=np.uint8)
    y = 40

    for _, row in df.iterrows():
        text = f"{row['Name']}  {row['Time']}"
        cv2.putText(img, text, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        y += 40

    cv2.imshow("Today's Attendance", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
