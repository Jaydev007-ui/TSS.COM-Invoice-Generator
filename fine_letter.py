import base64
import io
from PIL import Image
from datetime import datetime

def generate_fine_letter(alert):
    emp = alert['details']
    
    # Convert employee photo to base64
    employee_image = Image.open(io.BytesIO(emp['photo']))
    buffered_employee = io.BytesIO()
    employee_image.save(buffered_employee, format="PNG")
    employee_img_str = base64.b64encode(buffered_employee.getvalue()).decode()

    # Convert spitting incident photo to base64
    incident_image = Image.fromarray(alert['image'])
    buffered_incident = io.BytesIO()
    incident_image.save(buffered_incident, format="PNG")
    incident_img_str = base64.b64encode(buffered_incident.getvalue()).decode()
    
    # Get current date and time
    dt = datetime.strptime(alert['timestamp'], "%Y-%m-%d %H:%M:%S")
    
    # HTML template with improved styling
    html_content = f"""
    <html>
    <head>
    <style>
        body {{ font-family: 'Arial', sans-serif; }}
        .letter-container {{
            border: 3px solid #e74c3c;
            border-radius: 15px;
            padding: 30px;
            max-width: 800px;
            margin: 20px auto;
            background: #f9f9f9;
        }}
        .header {{
            text-align: center;
            color: #e74c3c;
            border-bottom: 2px solid #e74c3c;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .logo {{
            width: 180px;
            margin-bottom: 15px;
        }}
        .section {{
            margin: 25px 0;
            padding: 15px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .signature-box {{
            margin-top: 40px;
            text-align: right;
            padding: 20px;
            border-top: 2px dashed #e74c3c;
        }}
        .fine-amount {{
            color: #e74c3c;
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            margin: 25px 0;
        }}
        .employee-photo {{
            border: 2px solid #e74c3c;
            border-radius: 8px;
            margin: 15px 0;
        }}
    </style>
    </head>
    <body>
        <div class="letter-container">
            <div class="header">
                <h1>🛡️ SPITSHIELD PRO</h1>
                <h3>Public Health Violation Notice</h3>
            </div>
            
            <div class="section">
                <h2>📅 Violation Details</h2>
                <p><strong>Date:</strong> {dt.strftime('%d %B %Y')}</p>
                <p><strong>Time:</strong> {dt.strftime('%I:%M %p')}</p>
                <p><strong>Location:</strong> Main Office Premises</p>
            </div>

            <div class="section">
                <h2>👤 Offender Information</h2>
                <img src="data:image/png;base64,{employee_img_str}" class="employee-photo" width="150">
                <p><strong>Name:</strong> {emp['name']}</p>
                <p><strong>Employee ID:</strong> {alert['emp_id']}</p>
                <p><strong>Contact:</strong> {emp['phone']}</p>
            </div>

            <div class="fine-amount">
                ₹500 FINE IMPOSED
            </div>

            <div class="section">
                <h2>⚖️ Violation Particulars</h2>
                <p>Violation Code: SS-102</p>
                <p>Article 15 of Public Health & Safety Act, 2018</p>
                <p>Match Confidence: {alert['similarity']*100:.2f}%</p>
                
                <!-- Added incident proof section -->
                <div style="margin-top: 20px;">
                    <img src="data:image/png;base64,{incident_img_str}" 
                         class="incident-proof"
                         alt="Spitting Incident Proof">
                    <p class="proof-caption">Spitting Incident Visual Proof</p>
                </div>
            </div>

            <div class="signature-box">
                <p>Authorized Signatory:</p>
                <img src="https://cdn-icons-png.flaticon.com/512/1496/1496034.png" width="120">
                <p>SpitShield Pro Enforcement Unit</p>
                <p>Date: {datetime.now().strftime('%d %B %Y')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

