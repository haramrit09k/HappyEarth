import streamlit as st
from PIL import Image
from clf import predict2
import time

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Dont Trash Me")
st.markdown("## Upload an image and I will tell you how you can reuse/recycle it (i.e. if it is recyclable :stuck_out_tongue:)\n\n\n<br></br>", unsafe_allow_html=True)

file_up = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "webp"])

plastic_bottles = {
    "name": "Plastic Bottle",
    "reusable":True,
    "use":"bird feeders, storage containers, stationary stands, etc.",
    "cfoot":"82.8 grams CO2 (for one 500ml plastic bottle)"
}

plastic_bags = {
    "name": "Plastic Bag",
    "reusable":True,
    "use":"storing objects while moving, substitute trash can liners.",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

can = {
    "name": "Soda Can",
    "reusable":True,
    "use":"To plant, to hold writing utensils",
    "cfoot":"142 grams CO2 (for one 355 ml aluminum soda can)"
}

cardboard = {
    "name": "Cardboard",
    "reusable":True,
    "use":"",
    "cfoot":"3.31 tonnes CO2 (for one tonne of cardboard)"
}

pizza_box = {
    "name": "Pizza Box",
    "reusable":True,
    "use":"",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

steel_container = {
    "name": "Steel Container",
    "reusable":True,
    "use":"",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

beer_bottle = {
    "name": "Beer Bottle",
    "reusable":True,
    "use":"",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

book = {
    "name": "Book",
    "reusable":True,
    "use":"Art crafts",
    "cfoot":"2-3 kg CO2 equivalent (for one book)"
}

egg = {
    "name": "Egg",
    "reusable":False,
    "use":"",
    "cfoot":"Unknown"
}

flower = {
    "name": "Flower",
    "reusable":False,
    "use":"",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

food_peels = {
    "name": "Food Peels",
    "reusable":False,
    "use":"",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

fruit = {
    "name": "Fruits",
    "reusable":False,
    "use":"",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

jute = {
    "name": "Jute",
    "reusable":False,
    "use":"",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

leaf = {
    "name": "Leaf",
    "reusable":False,
    "use":"",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

meat = {
    "name": "Meat",
    "reusable":False,
    "use":"",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

newspaper = {
    "name": "Newspaper",
    "reusable":True,
    "use":"Art crafts",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

plants = {
    "name": "Plants",
    "reusable":False,
    "use":"",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

spoilt_food = {
    "name": "Spoilt Food",
    "reusable":False,
    "use":"",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}

thermocol = {
    "name": "Thermocol",
    "reusable":True,
    "use":"",
    "cfoot":"33 grams CO2 (for one average plastic grocery bag)"
}


if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    # st.write("Just a second...")
    labels = predict2(file_up)

    st.write(labels[0][1])
    with st.spinner('Processing...'):
        time.sleep(5)
    # print out the top prediction label with score
    if labels[0][1] < 70:
        st.markdown("<div style='color: blue; font-size: 40px;'> I am not sure what this object is! </div>", unsafe_allow_html=True)
        st.markdown("<div style='color: red; font-size: 20px;'> I am not smart enough... *cries in corner* </div>", unsafe_allow_html=True)
        st.image('media/crying.gif')
    else:
        res = {}
        if labels[0][0] == 'can':
            res = can
        elif labels[0][0] == 'plastic-bottle':
            res = plastic_bottles
        elif labels[0][0] == 'plastic-bag':
            res = plastic_bags
        elif labels[0][0] == 'beer-bottle':
            res = beer_bottle
        elif labels[0][0] == 'cardboard':
            res = cardboard
        elif labels[0][0] == 'book':
            res = book
        elif labels[0][0] == 'egg':
            res = egg
        elif labels[0][0] == 'flower':
            res = flower
        elif labels[0][0] == 'food-peels':
            res = food_peels
        elif labels[0][0] == 'fruit':
            res = fruit
        elif labels[0][0] == 'jute':
            res = jute
        elif labels[0][0] == 'leaf':
            res = meat
        elif labels[0][0] == 'newspaper':
            res = newspaper
        elif labels[0][0] == 'pizza-box':
            res = pizza_box
        elif labels[0][0] == 'plant':
            res = plants
        elif labels[0][0] == 'spoilt-food':
            res = spoilt_food
        elif labels[0][0] == 'steel-container':
            res = steel_container
        elif labels[0][0] == 'thermocol':
            res = thermocol
        st.write("This looks like...")
        st.markdown("<div style='color: blue; font-size: 40px;'>"+res.get('name')+"</div>", unsafe_allow_html=True)
        st.markdown("<div style='color: green; font-size: 20px;'> Reusable? : "+str(res.get('reusable'))+"</div>", unsafe_allow_html=True)
        if res.get('reusable') == True:
            st.markdown("<div style='color: green; font-size: 20px;'> How can I reuse it?: "+str(res.get('use'))+"</div>", unsafe_allow_html=True)
            st.markdown("<div style='color: green; font-size: 20px;'> Carbon footprint: "+str(res.get('cfoot'))+"</div>", unsafe_allow_html=True)
    
    
