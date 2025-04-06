from dotenv import load_dotenv
import os
import psycopg2
from tabulate import tabulate

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

def get_db_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None
    
    
def get_users(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    print(tabulate(users,column_names, tablefmt="psql"))
    return users


def main():
    conn = get_db_connection()
    if conn:
        get_users(conn)
        conn.close()
    else:
        print("Failed to connect to the database.")

if __name__ == "__main__":
    main()

