import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

st.set_page_config(page_title="Text Summarization")

model_name = "facebook/bart-large-xsum"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

st.title("Text Summarization using LLM")

text = st.text_area("Enter News Article")

if st.button("Summarize"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=130,
            min_length=20,
            num_beams=4
        )

        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        st.subheader("Summary")
        st.write(summary)
