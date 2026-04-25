import datetime
from fpdf import FPDF


def generate_chat_pdf(messages: list, doc_names: list) -> bytes:
    """Export conversation as a PDF report."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, "RAG Chatbot - Conversation Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}", ln=True, align="C")
    pdf.ln(4)

    # Documents
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 9, "Documents Analyzed", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    for name in doc_names:
        pdf.cell(6)
        pdf.cell(0, 7, f"- {''.join(c if ord(c) < 128 else '?' for c in name)}", ln=True)
    pdf.ln(6)

    # Conversation
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 9, "Conversation", ln=True, fill=True)
    pdf.ln(4)

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", "")
        msg_type = msg.get("type", "text")

        if role == "user":
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 100, 200)
            label = "You (Voice)" if msg_type == "voice" else ("You (Image)" if msg_type == "image" else "You")
            pdf.cell(0, 7, label, ln=True)
        else:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 150, 80)
            pdf.cell(0, 7, "Assistant", ln=True)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 6, "".join(c if ord(c) < 128 else "?" for c in content))

        if sources:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(130, 130, 130)
            pdf.cell(0, 6, "".join(c if ord(c) < 128 else "?" for c in sources), ln=True)

        pdf.ln(4)
        if i < len(messages) - 1:
            pdf.set_draw_color(220, 220, 220)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)

    return pdf.output()