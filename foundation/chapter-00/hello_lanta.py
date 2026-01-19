#!/usr/bin/env python3
"""
Hello LANTA - ตัวอย่างแรกสำหรับทดสอบการเชื่อมต่อ
Chapter 0: HPC 101

รันได้ทั้งบน LANTA และเครื่องส่วนตัว
"""

import os
import platform
import socket
from datetime import datetime


def get_system_info():
    """รวบรวมข้อมูลระบบ"""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now().isoformat(),
    }

    # ตรวจสอบว่าอยู่บน LANTA หรือไม่
    info["is_lanta"] = "lanta" in info["hostname"].lower()

    # ตรวจสอบ SLURM environment
    slurm_vars = ["SLURM_JOB_ID", "SLURM_NODELIST", "SLURM_NTASKS"]
    info["slurm"] = {var: os.environ.get(var, "N/A") for var in slurm_vars}

    return info


def main():
    print("=" * 60)
    print("🚀 สวัสดี LANTA! - Hello LANTA!")
    print("=" * 60)

    info = get_system_info()

    print(f"\n📍 Hostname: {info['hostname']}")
    print(f"💻 Platform: {info['platform']}")
    print(f"🔧 Processor: {info['processor']}")
    print(f"🐍 Python: {info['python_version']}")
    print(f"🕐 Timestamp: {info['timestamp']}")

    if info["is_lanta"]:
        print("\n✅ คุณกำลังรันบนระบบ LANTA!")
        print("\nSLURM Job Information:")
        for key, value in info["slurm"].items():
            print(f"  {key}: {value}")
    else:
        print("\n📌 คุณกำลังรันบนเครื่องส่วนตัว")
        print("   ลองรันบน LANTA ด้วยคำสั่ง: sbatch hello_lanta.sbatch")

    print("\n" + "=" * 60)
    print("🎉 ยินดีต้อนรับสู่ HPC Ignite Hands-On Labs!")
    print("=" * 60)


if __name__ == "__main__":
    main()
