from legal_qa import api


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["documents_indexed"] == 12  # 6 articles x (ar, en)


def test_ask_contract(client):
    r = client.post("/ask", json={"question": "هل يسأل من استعمل حقه استعمالا مشروعا عن الضرر؟"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "sources"}
    assert isinstance(body["answer"], str) and body["answer"]
    assert body["sources"] and all(isinstance(s, str) for s in body["sources"])
    assert "Article 4" in body["sources"] or "المادة 4" in body["sources"]


def test_ask_english(client):
    body = client.post("/ask", json={"question": "When does legal personality of a natural person begin?"}).json()
    assert any("6" in s for s in body["sources"])


def test_empty_question_422(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_blank_question_422(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_missing_question_422(client):
    assert client.post("/ask", json={}).status_code == 422


def test_refusal_has_no_sources(client):
    api.state["llm"].reply = api.REFUSAL_AR
    body = client.post("/ask", json={"question": "ما عاصمة فرنسا؟"}).json()
    assert body["sources"] == []
    assert body["answer"] == api.REFUSAL_AR


def test_repealed_article_hidden_by_default(client):
    hits = client.post("/search", json={"question": "الاختصاص القضائي القوانين الخاصة"}).json()
    assert 3 not in [h["article_number"] for h in hits]
    hits = client.post("/search", json={"question": "المادة الملغاة الاختصاص القضائي"}).json()
    assert 3 in [h["article_number"] for h in hits]


def test_stream_events(client):
    r = client.post("/ask/stream", json={"question": "الشخص الطبيعي تبدأ شخصيته متى؟"})
    assert r.status_code == 200
    text = r.text
    assert text.index("event: sources") < text.index("event: token") < text.index("event: done")


def test_llm_missing_returns_503(client):
    api.state.pop("llm")
    assert client.post("/ask", json={"question": "سؤال"}).status_code == 503
