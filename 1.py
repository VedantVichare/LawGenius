from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "nlpaueb/legal-bert-base-uncased"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./legalbert_model")
tokenizer.save_pretrained("./legalbert_model")