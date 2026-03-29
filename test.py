import requests

while True:
    title = input("Enter title: ")

    if title.lower() == "exit":
        break

    res = requests.get(
        "https://deployed-reo-ai-cf3syvobgnqkkucgnxdgmj.streamlit.app/predict",
        params={"title": title}
    )

    print(res.json())
