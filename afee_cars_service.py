import sqlite3
import json
from datetime import datetime, timedelta
import uuid

# Initialize SQLite database
def init_database():
    conn = sqlite3.connect('afee_cars.db')
    c = conn.cursor()
    
    # Create customers table
    c.execute('''CREATE TABLE IF NOT EXISTS customers
                 (id TEXT PRIMARY KEY,
                  full_name TEXT NOT NULL,
                  phone TEXT,
                  car_make TEXT,
                  car_model TEXT,
                  fuel_type TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create appointments table
    c.execute('''CREATE TABLE IF NOT EXISTS appointments
                 (id TEXT PRIMARY KEY,
                  customer_id TEXT,
                  service_type TEXT,
                  appointment_date DATE,
                  status TEXT DEFAULT 'scheduled',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (customer_id) REFERENCES customers (id))''')
    
    conn.commit()
    return conn

# Customer management
def find_or_create_customer(conn, name, phone="", car_make="", car_model="", fuel_type=""):
    c = conn.cursor()
    
    # Try to find existing customer by name and phone
    c.execute("SELECT id FROM customers WHERE full_name = ? AND phone = ?", 
              (name, phone))
    result = c.fetchone()
    
    if result:
        return result[0]  # Return existing customer ID
    else:
        # Create new customer
        customer_id = str(uuid.uuid4())
        c.execute('''INSERT INTO customers (id, full_name, phone, car_make, car_model, fuel_type)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (customer_id, name, phone, car_make, car_model, fuel_type))
        conn.commit()
        return customer_id

def get_customer_details(conn, customer_id):
    c = conn.cursor()
    c.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
    return c.fetchone()

# Appointment management
def create_appointment(conn, customer_id, service_type, appointment_date):
    c = conn.cursor()
    appointment_id = str(uuid.uuid4())
    
    c.execute('''INSERT INTO appointments (id, customer_id, service_type, appointment_date)
                 VALUES (?, ?, ?, ?)''',
              (appointment_id, customer_id, service_type, appointment_date))
    conn.commit()
    return appointment_id

def get_available_dates():
    # Generate available dates (next 7 days, excluding weekends)
    today = datetime.now()
    dates = []
    days_added = 0
    
    while len(dates) < 7:
        date = today + timedelta(days=days_added)
        if date.weekday() < 5:  # Monday = 0, Sunday = 6
            dates.append(date.strftime('%Y-%m-%d'))
        days_added += 1
    
    return dates

# Main conversation flow
def main():
    print("Initializing Afee Cars Service...")
    conn = init_database()
    
    print("\n" + "="*50)
    print("Afee Cars Service - AI Assistant")
    print("="*50)
    
    # Step 1: Greeting
    print("\nAI: ഹലോ! ഇത് 'അഫി കാർ സർവീസ്' ആണ്. നിങ്ങൾക്ക് എങ്ങനെ സഹായിക്കാമെന്ന് പറയൂ.")
    
    # Step 2: Get customer name
    while True:
        name = input("\nYour full name: ").strip()
        if name:
            break
        print("AI: ദയവായി സാധുവായ ഒരു പേര് നൽകുക.")
    
    # Check if customer exists
    c = conn.cursor()
    c.execute("SELECT * FROM customers WHERE full_name = ?", (name,))
    existing_customer = c.fetchone()
    
    if existing_customer:
        customer_id = existing_customer[0]
        print(f"AI: ഹായ് {name}! നിങ്ങളുടെ വിവരങ്ങൾ ഞങ്ങളുടെ റെക്കോർഡിൽ ഉണ്ട്. നിങ്ങളുടെ കാർ {existing_customer[3]} {existing_customer[4]} ({existing_customer[5]}) ആണല്ലേ?")
    else:
        print("AI: നിങ്ങളുടെ വിവരങ്ങൾ ഞങ്ങളുടെ റെക്കോർഡിൽ കണ്ടെത്താനായില്ല. ദയവായി കുറച്ച് വിവരങ്ങൾ നൽകുക.")
        
        phone = input("AI: നിങ്ങളുടെ ഫോൺ നമ്പർ: ").strip()
        car_make = input("AI: കാറിന്റെ മേക്ക് (ഉദാ: Hyundai, Maruti): ").strip()
        car_model = input("AI: കാറിന്റെ മോഡൽ: ").strip()
        fuel_type = input("AI: ഇന്ധന തരം (Petrol/Diesel): ").strip().lower()
        
        customer_id = find_or_create_customer(conn, name, phone, car_make, car_model, fuel_type)
    
    # Step 3: Get service details
    print("\nAI: എന്ത് തരത്തിലുള്ള സർവീസ് ആണ് ആവശ്യം? (ഉദാ: സാധാരണ സർവീസ്, ബ്രേക്ക് സർവീസ്, എയർ കണ്ടീഷണർ സർവീസ്)")
    service_type = input("Service needed: ").strip()
    
    # Step 4: Show available dates
    print("\nAI: ഈ താഴെയുള്ള തീയതികളിൽ ഏതെങ്കിലും തിരഞ്ഞെടുക്കുക:")
    available_dates = get_available_dates()
    
    for i, date in enumerate(available_dates, 1):
        print(f"{i}. {date}")
    
    # Step 5: Get preferred date
    while True:
        try:
            choice = int(input("\nAI: ദയവായി നിങ്ങൾ തിരഞ്ഞെടുക്കുന്ന ഒരു നമ്പർ നൽകുക (1-7): "))
            if 1 <= choice <= len(available_dates):
                selected_date = available_dates[choice-1]
                break
            print("AI: ദയവായി 1 മുതൽ 7 വരെയുള്ള ഒരു സംഖ്യ നൽകുക.")
        except ValueError:
            print("AI: താക്കീത്: സാധുവായ ഒരു നമ്പർ നൽകുക.")
    
    # Step 6: Confirm booking
    print(f"\nAI: നിങ്ങളുടെ ബുക്കിംഗ് വിശദാംശങ്ങൾ:")
    print(f"- പേര്: {name}")
    if existing_customer:
        print(f"- കാർ: {existing_customer[3]} {existing_customer[4]} ({existing_customer[5]})")
    print(f"- സർവീസ്: {service_type}")
    print(f"- തീയതി: {selected_date}")
    
    confirm = input("\nAI: ഈ വിവരങ്ങൾ ശരിയാണോ? (അതെ/ഇല്ല): ").strip().lower()
    
    if confirm in ['yes', 'y', 'അതെ', 'അതേ', 'ശരി']:
        # Create appointment
        appointment_id = create_appointment(conn, customer_id, service_type, selected_date)
        print(f"\nAI: നന്ദി! നിങ്ങളുടെ അപ്പോയിന്റ്മെന്റ് ബുക്ക് ചെയ്തിരിക്കുന്നു. നിങ്ങളുടെ അപ്പോയിന്റ്മെന്റ് ഐഡി: {appointment_id}")
        print(f"AI: താങ്കൾ തിരഞ്ഞെടുത്ത തീയതി: {selected_date}")
    else:
        print("\nAI: ബുക്കിംഗ് റദ്ദാക്കി. വീണ്ടും ശ്രമിക്കാം.")
    
    # Close connection
    conn.close()
    print("\nAI: നന്ദി! 'അഫി കാർ സർവീസ്' തിരഞ്ഞെടുക്കുന്നതിന് നന്ദി. നിങ്ങൾക്ക് നല്ല ഒരു ദിവസം!")

if __name__ == "__main__":
    main()
