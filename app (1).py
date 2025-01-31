import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, auth
import tempfile
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("path/to/your/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def main():
    st.set_page_config(page_title="TSS.COM Invoice Generator", page_icon=":receipt:", layout="wide")
    st.title("TSS.COM Invoice Generator")

    if 'user' not in st.session_state:
        st.session_state.user = None

    if st.session_state.user is None:
        show_login_signup()
    else:
        show_main_app()

def show_login_signup():
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                user = auth.get_user_by_email(f"{username}@tss.com")
                # In a real app, you'd verify the password here
                st.session_state.user = user
                st.success("Logged in successfully!")
                st.experimental_rerun()
            except:
                st.error("Invalid credentials")

    with tab2:
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Sign Up"):
            try:
                user = auth.create_user(
                    email=f"{new_username}@tss.com",
                    password=new_password
                )
                st.success("Account created successfully! Please log in.")
            except:
                st.error("Failed to create account")

def show_main_app():
    st.sidebar.title(f"Welcome, {st.session_state.user.email.split('@')[0]}!")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.experimental_rerun()

    tab1, tab2, tab3 = st.tabs(["Profile", "Generate Invoice", "View Invoices"])

    with tab1:
        show_profile()

    with tab2:
        generate_invoice()

    with tab3:
        view_invoices()

def show_profile():
    st.header("Profile")
    user_ref = db.collection('users').document(st.session_state.user.uid)
    user_data = user_ref.get().to_dict()

    if user_data is None:
        user_data = {}

    name = st.text_input("Name", value=user_data.get('name', ''))
    department = st.text_input("Department", value=user_data.get('department', ''))
    photo = st.file_uploader("Profile Photo", type=['png', 'jpg', 'jpeg'])

    if st.button("Save Profile"):
        update_data = {
            'name': name,
            'department': department
        }
        if photo:
            # In a real app, you'd upload this to cloud storage and save the URL
            update_data['photo_url'] = "photo_url_placeholder"

        user_ref.set(update_data, merge=True)
        st.success("Profile updated successfully!")

def generate_invoice():
    st.header("Generate Invoice")
    
    # Invoice details
    invoice_number = st.text_input("Invoice Number")
    client_name = st.text_input("Client Name")
    invoice_date = st.date_input("Invoice Date")
    due_date = st.date_input("Due Date")

    # Invoice items
    items = []
    for i in range(5):  # Allow up to 5 items
        col1, col2, col3 = st.columns(3)
        with col1:
            item = st.text_input(f"Item {i+1}")
        with col2:
            quantity = st.number_input(f"Quantity {i+1}", min_value=0, value=0)
        with col3:
            price = st.number_input(f"Price {i+1}", min_value=0.0, value=0.0)
        if item and quantity > 0 and price > 0:
            items.append({"item": item, "quantity": quantity, "price": price})

    if st.button("Generate Invoice"):
        invoice_data = {
            "invoice_number": invoice_number,
            "client_name": client_name,
            "invoice_date": invoice_date.isoformat(),
            "due_date": due_date.isoformat(),
            "items": items,
            "created_by": st.session_state.user.uid,
            "created_at": datetime.now().isoformat()
        }
        db.collection('invoices').add(invoice_data)
        st.success("Invoice generated successfully!")
        
        # Generate PDF
        pdf_buffer = generate_pdf(invoice_data)
        st.download_button(
            label="Download Invoice PDF",
            data=pdf_buffer,
            file_name=f"invoice_{invoice_number}.pdf",
            mime="application/pdf"
        )

def view_invoices():
    st.header("View Invoices")
    invoices = db.collection('invoices').get()

    for invoice in invoices:
        invoice_data = invoice.to_dict()
        st.subheader(f"Invoice {invoice_data['invoice_number']}")
        st.write(f"Client: {invoice_data['client_name']}")
        st.write(f"Date: {invoice_data['invoice_date']}")
        st.write(f"Due Date: {invoice_data['due_date']}")
        st.write("Items:")
        for item in invoice_data['items']:
            st.write(f"- {item['item']}: {item['quantity']} x ${item['price']}")
        
        # Generate PDF for viewing
        pdf_buffer = generate_pdf(invoice_data)
        st.download_button(
            label="Download Invoice PDF",
            data=pdf_buffer,
            file_name=f"invoice_{invoice_data['invoice_number']}.pdf",
            mime="application/pdf"
        )

def generate_pdf(invoice_data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add logo
    # c.drawImage("path/to/logo.png", 50, height - 100, width=100, height=50)

    # Add invoice details
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 150, f"Invoice #{invoice_data['invoice_number']}")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 170, f"Client: {invoice_data['client_name']}")
    c.drawString(50, height - 190, f"Date: {invoice_data['invoice_date']}")
    c.drawString(50, height - 210, f"Due Date: {invoice_data['due_date']}")

    # Add items
    y = height - 250
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Item")
    c.drawString(250, y, "Quantity")
    c.drawString(350, y, "Price")
    c.drawString(450, y, "Total")
    y -= 20

    total = 0
    for item in invoice_data['items']:
        c.setFont("Helvetica", 12)
        c.drawString(50, y, item['item'])
        c.drawString(250, y, str(item['quantity']))
        c.drawString(350, y, f"${item['price']:.2f}")
        item_total = item['quantity'] * item['price']
        c.drawString(450, y, f"${item_total:.2f}")
        total += item_total
        y -= 20

    # Add total
    c.setFont("Helvetica-Bold", 12)
    c.drawString(350, y - 20, "Total:")
    c.drawString(450, y - 20, f"${total:.2f}")

    c.save()
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    main()

