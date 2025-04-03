from src.rag_pipeline import ask_question

while True:
    query = input("💬 Ask a question (type 'exit' to quit): ")
    if query.lower() == "exit":
        print("👋 Exiting...")
        break
    try:
        response = ask_question(query)
        print("🤖 Answer:", response)
    except Exception as e:
        print("❌ Error:", str(e))