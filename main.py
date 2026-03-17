from NLP.pipeline import SentimentPipeline

def main():

    text = input("Introduce texto:\n")

    pipeline = SentimentPipeline()

    result = pipeline.process(text)

    print("\nTexto limpio:", result["clean_text"])
    print("Tokens:", result["tokens"])
    print("Emoción:", result["emotion"])
    print("Conteo:", result["counter"])

if __name__ == "__main__":
    main()