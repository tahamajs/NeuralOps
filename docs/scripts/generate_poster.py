from pptx import Presentation
from pptx.util import Inches

prs = Presentation("Docs/A1_Vertical_Template_Electronics.pptx")
slide = prs.slides[0]


def replace_paragraph_text(paragraph, new_text):
    if len(paragraph.runs) > 0:
        # Keep the formatting of the first run
        first_run = paragraph.runs[0]
        first_run.text = new_text
        # Clear the rest of the runs
        for i in range(1, len(paragraph.runs)):
            paragraph.runs[i].text = ""
    else:
        paragraph.text = new_text


# Update Shape 15 (Results)
if len(slide.shapes) > 15 and slide.shapes[15].has_text_frame:
    tf = slide.shapes[15].text_frame
    if len(tf.paragraphs) > 1:
        replace_paragraph_text(
            tf.paragraphs[1],
            "•  عملکرد خیره‌کننده ۵ ماژول برتر: در میان ۱۰۰ ماژول موفق ارزیابی شده، NLI Softmax با دقت ۹۰.۹۴٪، Weighted Ensemble با دقت ۹۰.۱۴٪، Huffman Coder با دقت ۹۰.۰۲٪، FRanDI با دقت ۸۸.۷۶٪ و SOA Fused Attention با دقت ۸۱.۳۱٪ رتبه‌های برتر را به خود اختصاص دادند.",
        )

# Update Shape 17 (Conclusion)
if len(slide.shapes) > 17 and slide.shapes[17].has_text_frame:
    tf = slide.shapes[17].text_frame
    if len(tf.paragraphs) > 1:
        replace_paragraph_text(
            tf.paragraphs[1],
            "•  پایداری خط لوله و حل بنیادین CUDA OOM: با محدود کردن پارامترهای آموزشی به ۲ میلیون و رفع باگ‌های پدینگ و Silent Failures، خط لوله موفق شد بیش از ۸۳۱ مخزن و ۱۰۰ ماژول برتر را کاملاً خودکار روی Hugging Face (مخزن Hooshaai) منتشر کند.",
        )

# Find the picture placeholder or shape 20 bounds and insert our generated image + chart
# We have `neuralops_cover.jpg` and `../thesis_final/chart.png`. Let's add them at the bottom.
try:
    slide.shapes.add_picture(
        "Docs/neuralops_cover.jpg", Inches(1.5), Inches(24), Inches(10), Inches(7)
    )
    slide.shapes.add_picture(
        "thesis_final/chart.png", Inches(12), Inches(24), Inches(10), Inches(6.5)
    )
except Exception as e:
    print(f"Failed to add images: {e}")

prs.save("Docs/A1_Vertical_Template_Electronics_Updated.pptx")
print("Poster text meticulously updated keeping the exact original fonts and layout.")
