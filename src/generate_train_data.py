import pandas as pd
import json

# Load your CSV data
df = pd.read_csv("./data/jan_sales.csv")

def generate_training_data(df):
    training_data = []
    
    for _, row in df.iterrows():
        # Total Sales
        training_data.append({
            "instruction": f"What were the total sales on {row['orderDate']}?",
            "input": "",
            "output": f"The total sales on {row['orderDate']} were ${row['orderAmountNet']:.2f}."
        })

        # Product Sales
        training_data.append({
            "instruction": f"How many units of {row['productName']} were sold on {row['orderDate']}?",
            "input": "",
            "output": f"A total of {row['quantity']} units of {row['productName']} were sold on {row['orderDate']}."
        })

        # Store Sales
        training_data.append({
            "instruction": f"What were the total sales at {row['storeName']} on {row['orderDate']}?",
            "input": "",
            "output": f"The total sales at {row['storeName']} on {row['orderDate']} were ${row['orderAmountNet']:.2f}."
        })

        # Brand Sales
        training_data.append({
            "instruction": f"What were the total sales for {row['brandName']} on {row['orderDate']}?",
            "input": "",
            "output": f"The total sales for {row['brandName']} on {row['orderDate']} were ${row['orderAmountNet']:.2f}."
        })

        # Discount Information
        training_data.append({
            "instruction": f"What was the discount given on {row['productName']}?",
            "input": "",
            "output": f"The discount on {row['productName']} was ${row['discountAmount']:.2f}."
        })

        # Most Used Payment Method
        training_data.append({
            "instruction": f"What was the most used payment method on {row['orderDate']}?",
            "input": "",
            "output": f"On {row['orderDate']}, the most used payment method was {row['paymentMethod']}."
        })

        # Most Sold Product on a Date
        training_data.append({
            "instruction": f"What was the most sold product on {row['orderDate']}?",
            "input": "",
            "output": f"The most sold product on {row['orderDate']} was {row['productName']} with {row['quantity']} units sold."
        })

        # Category Sales
        training_data.append({
            "instruction": f"What were the total sales in the {row['categoryName']} category on {row['orderDate']}?",
            "input": "",
            "output": f"The total sales in the {row['categoryName']} category on {row['orderDate']} were ${row['orderAmountNet']:.2f}."
        })

        # Subcategory Sales
        training_data.append({
            "instruction": f"What were the total sales in the {row['subCategoryOf']} subcategory on {row['orderDate']}?",
            "input": "",
            "output": f"The total sales in the {row['subCategoryOf']} subcategory on {row['orderDate']} were ${row['orderAmountNet']:.2f}."
        })

        # Frequently Bought Together (Per Product)
        training_data.append({
            "instruction": f"What products are frequently bought with {row['productName']}?",
            "input": "",
            "output": f"Customers who bought {row['productName']} also frequently bought related products."
        })

        # Frequently Bought Together (Overall)
        training_data.append({
            "instruction": "What are products that are frequently bought together?",
            "input": "",
            "output": "Frequently bought together products include various product combinations based on past sales trends."
        })

    return training_data

# Generate fine-tuning dataset
qa_data = generate_training_data(df)

# Save as JSON
with open("./sales_dataset.json", "w") as f:
    json.dump(qa_data, f, indent=4)

print("Sales dataset created successfully")
