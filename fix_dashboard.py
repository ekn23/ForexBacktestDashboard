"""
Quick fix for dashboard timestamp issues
Ensures clean zero baselines display properly
"""

import os
import pandas as pd

def fix_app_timestamps():
    """Fix the timestamp comparison issues in app.py"""
    
    # Read the current app.py file
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Fix the timestamp comparison issues
    old_pattern = "df = df[df['datetime'] >= start_date]"
    new_pattern = """start_dt = pd.to_datetime(start_date, utc=True)
                            df = df[df['datetime'] >= start_dt]"""
    
    content = content.replace(old_pattern, new_pattern)
    
    old_pattern2 = "df = df[df['datetime'] <= end_date]"
    new_pattern2 = """end_dt = pd.to_datetime(end_date, utc=True)
                            df = df[df['datetime'] <= end_dt]"""
    
    content = content.replace(old_pattern2, new_pattern2)
    
    # Also fix the datetime parsing to be consistent
    old_datetime = "df['datetime'] = pd.to_datetime(df['Local time'])"
    new_datetime = "df['datetime'] = pd.to_datetime(df['Local time'], utc=True)"
    
    content = content.replace(old_datetime, new_datetime)
    
    # Write the fixed content back
    with open('app.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed timestamp comparison issues!")
    print("Your dashboard should now display clean zero baselines properly")

if __name__ == "__main__":
    fix_app_timestamps()