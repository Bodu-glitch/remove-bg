import subprocess
import time
import sys

def main():
    script_to_run = "test_analyze.py"
    print(f"Bắt đầu trình quản lý tiến trình cho: {script_to_run}")

    while True:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Đang khởi động {script_to_run}...")
        
        # Khởi chạy script
        process = subprocess.Popen([sys.executable, script_to_run])
        
        # Chờ script chạy xong
        process.wait()
        
        # Lấy mã lỗi trả về (0 là thành công, khác 0 là có lỗi văng ra)
        return_code = process.returncode
        
        if return_code == 0:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Quá trình hoàn tất 100%! Không còn lỗi nào.")
            break
        else:
            print(f"\n[CẢNH BÁO] Process bị sập hoặc trả về mã lỗi (Return code: {return_code}).")
            print("Đang tiến hành tự động khởi động lại sau 5 giây...")
            time.sleep(5)

if __name__ == "__main__":
    main()
