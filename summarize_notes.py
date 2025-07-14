from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

def summarize_notes(text):
    prompt = f"<s>Summarize the following clinical note:\n{text}\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    
    if "Summary:" in summary:
        summary = summary.split("Summary:")[-1].strip()
    return summary
