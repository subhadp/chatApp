import spacy
nlp = spacy.load("en_core_web_sm")

review_text = """
I recently purchased the new Apple iPhone 15 Pro Max after using the iPhone 13 for a couple of years, 
and I have to say that the improvements are remarkable. The A17 Bionic chip makes everything feel much smoother, 
and the battery life is noticeably better. The camera system is incredible, especially with the new 48MP sensor, 
making my photos look sharp even in low light. 

Along with the iPhone, I bought the AirPods Pro (2nd Generation), and they are a perfect match for the phone. 
The noise cancellation on the AirPods is fantastic, and the spatial audio makes music sound more immersive than ever.

I also considered the Samsung Galaxy S23 Ultra before deciding on the iPhone, and although the Galaxy is a fantastic device, 
I felt more comfortable sticking with Apple's ecosystem. My MacBook Pro 2021 and iPad Pro work seamlessly with the new iPhone. 
However, I did buy a Samsung Galaxy Tab S8 for my work meetings, and it has been a great productivity tool. 
The large screen and multi-window feature make it easy to handle several tasks at once.

For my fitness routine, I opted for the Apple Watch Ultra, and it's an absolute game-changer. The fitness tracking, 
heart rate monitor, and GPS accuracy are on point, and it integrates well with the Health app on my iPhone. 
I've used a Fitbit Versa 3 before, but the Apple Watch is in a league of its own. 

Overall, I'm incredibly happy with my purchases, and the entire Apple ecosystem just works so well together. 
I also picked up a Logitech MX Master 3S mouse, which is incredibly comfortable for daily use alongside my MacBook.
"""

review_text1 = """
I recently bought the new Apple iPhone 15, and I must say it's a game-changer!
The camera quality is amazing, and the battery life lasts all day.
I am also considering the new Samsung Galaxy Watch for fitness tracking.
"""

doc = nlp(review_text)

product_name = []

for ent in doc.ents:
    if ent.label_ in ["ORG", "PRODUCT", "GPE"]:
        product_name.append(ent.text)

print("Product Names Found:", product_name)