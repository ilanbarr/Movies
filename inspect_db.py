import sqlite3
import os

def inspect_database():
    db_path = 'db.sqlite'
    
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' does not exist.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables in the database:")
        for table in tables:
            print(table[0])

        # Show contents of the Preference table
        print("\nContents of the Preference table:")
        cursor.execute("SELECT * FROM Preference;")
        preferences = cursor.fetchall()
        for row in preferences:
            print(row)

        # Show contents of the User table
        print("\nContents of the User table:")
        cursor.execute("SELECT * FROM User;")
        users = cursor.fetchall()
        for row in users:
            print(row)

        conn.close()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

if __name__ == "__main__":
    inspect_database()
