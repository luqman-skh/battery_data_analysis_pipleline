def calc_similarity(doc1, doc2):
    model = SentenceTRansformer("all-MiniLM")

    embed1 = model.encode(doc1)
    embed2 = model.encode(doc2)

    similarity = cosine_similarity([embed1], [embed2])[0][0]

    return similarity