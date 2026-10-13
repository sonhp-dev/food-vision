import json
import os
import re
import time
from typing import Optional

import pandas as pd
import streamlit as st

from PIL import Image
from dotenv import load_dotenv


# ============================================================
# GEMINI
# ============================================================

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


# ============================================================
# CẤU HÌNH
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"

FOOD_CSV_PATH = "food.csv"
CONFIG_PATH = "config.json"


# ============================================================
# TÊN TIẾNG VIỆT
# ============================================================

VIETNAMESE_NAMES = {

    "apple": "táo",
    "banana": "chuối",
    "beetroot": "củ dền",
    "bell pepper": "ớt chuông",
    "cabbage": "bắp cải",
    "capsicum": "ớt chuông",
    "carrot": "cà rốt",
    "cauliflower": "súp lơ trắng",
    "chilli pepper": "ớt",
    "corn": "ngô",
    "cucumber": "dưa chuột",
    "eggplant": "cà tím",
    "garlic": "tỏi",
    "ginger": "gừng",
    "grape": "nho",
    "green grape": "nho xanh",
    "jalepeno": "ớt jalapeño",
    "kiwi": "kiwi",
    "lemon": "chanh vàng",
    "lettuce": "xà lách",
    "mango": "xoài",
    "onion": "hành tây",
    "orange": "cam",
    "paprika": "ớt paprika",
    "pear": "lê",
    "peas": "đậu Hà Lan",
    "pineapple": "dứa",
    "pomegranate": "lựu",
    "potato": "khoai tây",
    "radish": "củ cải",
    "soy beans": "đậu nành",
    "spinach": "rau bina",
    "sweet corn": "ngô ngọt",
    "sweet potato": "khoai lang",
    "tomato": "cà chua",
    "cherry tomato": "cà chua bi",
    "turnip": "củ cải trắng",
    "watermelon": "dưa hấu",

    # Các loại Gemini có thể nhận diện thêm
    "fennel": "thì là",
    "grapefruit": "bưởi",
    "celery": "cần tây",
    "leek": "tỏi tây",
    "pumpkin": "bí đỏ",
    "broccoli": "súp lơ xanh",
    "avocado": "bơ",
    "asparagus": "măng tây",
    "parsley": "ngò tây",
    "strawberry": "dâu tây",
    "peach": "đào",
    "cherry": "anh đào"
}


def display_food_name(name: str) -> str:

    english_name = str(name).strip()

    vietnamese_name = VIETNAMESE_NAMES.get(
        english_name.lower()
    )

    if vietnamese_name:

        return (
            f"{english_name} "
            f"({vietnamese_name})"
        )

    # Nếu chưa có bản dịch thì giữ tiếng Anh
    return english_name


# ============================================================
# API KEY
# ============================================================

def get_streamlit_secret(
    secret_name: str
) -> Optional[str]:

    try:
        return st.secrets[secret_name]
    except Exception:
        return None


def load_api_key() -> Optional[str]:

    load_dotenv()

    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or get_streamlit_secret("GEMINI_API_KEY")
        or get_streamlit_secret("GOOGLE_API_KEY")
    )


# ============================================================
# LOAD CONFIG
# ============================================================

@st.cache_data
def load_config() -> dict:

    if not os.path.exists(CONFIG_PATH):

        return {
            "initial_bot_message":
                "Chào bạn! Mình là FoodBot 🥗. "
                "Bạn có thể tải ảnh rau củ quả lên "
                "để mình nhận diện, xem dinh dưỡng "
                "và gợi ý món ăn.",

            "assistant_name":
                "FoodBot",

            "functions": [
                "nhận diện rau củ quả",
                "phân tích dinh dưỡng",
                "gợi ý món ăn",
                "giải đáp câu hỏi về thực phẩm"
            ],

            "out_of_scope_message":
                "Xin lỗi, mình chỉ hỗ trợ các câu hỏi "
                "liên quan đến rau củ quả, thực phẩm, "
                "dinh dưỡng và món ăn."
        }

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# CHUẨN HÓA TÊN
# ============================================================

def normalize_food_name(name: str) -> str:

    name = str(name).lower().strip()

    # Bỏ ký tự đặc biệt
    name = re.sub(
        r"[^a-z0-9\s]",
        "",
        name
    )

    replacements = {
        "tomatoes": "tomato",
        "potatoes": "potato",
        "carrots": "carrot",
        "onions": "onion",
        "grapes": "grape",
        "raddish": "radish",
        "jalapeno": "jalepeno",
        "sweetcorn": "sweet corn",
        "sweetpotato": "sweet potato",
        "soybean": "soy beans",
        "soybeans": "soy beans",
        "capsicum": "bell pepper",
        "bell peppers": "bell pepper"
    }

    return replacements.get(
        name,
        name
    )


# ============================================================
# LOAD FOOD.CSV
# ============================================================

@st.cache_data
def load_food_csv() -> pd.DataFrame:

    df = pd.read_csv(
        FOOD_CSV_PATH,
        encoding="utf-8"
    )

    required_columns = [
        "Food Items",
        "Energy kcal",
        "Carbs",
        "Protein(g)",
        "Fat(g)",
        "Freesugar(g)",
        "Fibre(g)",
        "Cholestrol(mg)",
        "Calcium(mg)"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "food.csv thiếu các cột: "
            + ", ".join(missing_columns)
        )

    df["Food_Normalized"] = (
        df["Food Items"]
        .astype(str)
        .apply(normalize_food_name)
    )

    return df


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def create_client(
    api_key: Optional[str]
):

    if not api_key:
        return None

    if genai is None:
        return None

    try:

        return genai.Client(
            api_key=api_key
        )

    except Exception:

        return None


# ============================================================
# PHÂN TÍCH ẢNH
# ============================================================

def analyze_image(
    client,
    image: Image.Image
):

    if client is None:

        raise RuntimeError(
            "Gemini API chưa được kết nối."
        )


    prompt = """
Bạn là AI nhận diện rau củ quả.

Hãy phân tích TOÀN BỘ hình ảnh và nhận diện tất cả
rau, củ, quả hoặc thực phẩm thực vật có thể nhìn thấy.

Với mỗi loại thực phẩm:

- name: tên thực phẩm bằng tiếng Anh
- quantity: số lượng ước tính
- estimated_weight_grams:
  tổng khối lượng ước tính của loại đó trong ảnh, tính bằng gram

Yêu cầu:
- Nếu có nhiều vật thể cùng loại, gộp thành một dòng.
- Chỉ nhận diện những thực phẩm thực sự nhìn thấy.
- Không bịa thêm thực phẩm.
- Tên phải bằng tiếng Anh.
- estimated_weight_grams là TỔNG khối lượng của loại đó.
- Chỉ trả về JSON.
- Không thêm Markdown.

Định dạng:

[
  {
    "name": "tomato",
    "quantity": "6",
    "estimated_weight_grams": 580
  },
  {
    "name": "potato",
    "quantity": "5",
    "estimated_weight_grams": 720
  }
]
"""


    last_error = None


    for attempt in range(3):

        try:

            response = client.models.generate_content(

                model=MODEL_NAME,

                contents=[
                    prompt,
                    image
                ]
            )

            text_result = (
                response.text.strip()
            )

            # Xóa ```json nếu Gemini thêm
            text_result = re.sub(
                r"```json\s*|\s*```",
                "",
                text_result
            ).strip()


            data = json.loads(
                text_result
            )


            # Gemini có thể trả list trực tiếp
            if isinstance(data, list):

                foods = data

            # Hoặc trả {"foods": [...]}
            elif isinstance(data, dict):

                foods = data.get(
                    "foods",
                    []
                )

            else:

                raise ValueError(
                    "JSON Gemini không đúng định dạng."
                )


            cleaned_foods = []


            for food in foods:

                if not isinstance(
                    food,
                    dict
                ):
                    continue


                name = str(
                    food.get(
                        "name",
                        ""
                    )
                ).strip()


                quantity = str(
                    food.get(
                        "quantity",
                        "1"
                    )
                ).strip()


                weight = food.get(
                    "estimated_weight_grams",
                    0
                )


                try:

                    weight = float(
                        weight
                    )

                except:

                    weight = 0


                if name:

                    cleaned_foods.append({

                        "name":
                            name,

                        "quantity":
                            quantity,

                        "estimated_weight_grams":
                            weight
                    })


            return cleaned_foods


        except Exception as e:

            last_error = e

            error_text = str(e)


            if "503" in error_text:

                if attempt < 2:

                    wait_time = 5 * (
                        2 ** attempt
                    )

                    time.sleep(
                        wait_time
                    )

            else:

                break


    raise RuntimeError(
        "Gemini không thể phân tích ảnh.\n"
        f"Chi tiết: {last_error}"
    )


# ============================================================
# TÌM THỰC PHẨM TRONG FOOD.CSV
# ============================================================

def find_food_data(
    food_name: str,
    df: pd.DataFrame
):

    target = normalize_food_name(
        food_name
    )


    # Exact match
    exact = df[
        df["Food_Normalized"]
        == target
    ]

    if len(exact) > 0:

        return exact.iloc[0]


    # Partial match
    partial = df[
        df["Food_Normalized"].str.contains(
            target,
            case=False,
            na=False
        )
    ]

    if len(partial) > 0:

        return partial.iloc[0]


    return None


# ============================================================
# TÍNH DINH DƯỠNG
# ============================================================

def calculate_nutrition(
    detected_foods,
    df
):

    nutrition_results = []


    for food in detected_foods:

        food_name = food["name"]

        weight = float(
            food["estimated_weight_grams"]
        )


        row = find_food_data(
            food_name,
            df
        )


        # Giữ tên tiếng Anh trong dữ liệu
        # để tra CSV và tính toán
        result = {

            "Food":
                food_name,

            "Quantity":
                food["quantity"],

            "Weight (g)":
                weight
        }


        if row is None:

            result["Energy kcal"] = None
            result["Carbs"] = None
            result["Protein(g)"] = None
            result["Fat(g)"] = None
            result["Freesugar(g)"] = None
            result["Fibre(g)"] = None
            result["Cholestrol(mg)"] = None
            result["Calcium(mg)"] = None


        else:

            ratio = (
                weight / 100
            )


            nutrition_columns = [
                "Energy kcal",
                "Carbs",
                "Protein(g)",
                "Fat(g)",
                "Freesugar(g)",
                "Fibre(g)",
                "Cholestrol(mg)",
                "Calcium(mg)"
            ]


            for col in nutrition_columns:

                value = pd.to_numeric(
                    row[col],
                    errors="coerce"
                )


                if pd.isna(value):

                    result[col] = None

                else:

                    result[col] = (
                        value * ratio
                    )


        nutrition_results.append(
            result
        )


    return pd.DataFrame(
        nutrition_results
    )


# ============================================================
# TỔNG DINH DƯỠNG
# ============================================================

def calculate_total_nutrition(
    nutrition_df
):

    columns = [

        "Energy kcal",
        "Carbs",
        "Protein(g)",
        "Fat(g)",
        "Freesugar(g)",
        "Fibre(g)",
        "Cholestrol(mg)",
        "Calcium(mg)"
    ]


    totals = {}


    for col in columns:

        totals[col] = pd.to_numeric(

            nutrition_df[col],

            errors="coerce"

        ).sum()


    return totals


# ============================================================
# DANH SÁCH MÓN ĂN
# ============================================================

RECIPES = {

    "Salad rau củ tổng hợp": {
        "ingredients": {
            "tomato",
            "cucumber",
            "lettuce",
            "carrot",
            "onion",
            "bell pepper",
            "spinach"
        }
    },


    "Súp rau củ": {
        "ingredients": {
            "potato",
            "carrot",
            "onion",
            "garlic",
            "tomato",
            "corn",
            "spinach",
            "cauliflower",
            "cabbage"
        }
    },


    "Rau củ xào": {
        "ingredients": {
            "carrot",
            "bell pepper",
            "onion",
            "garlic",
            "spinach",
            "peas",
            "cabbage",
            "cauliflower"
        }
    },


    "Rau củ nướng": {
        "ingredients": {
            "potato",
            "carrot",
            "onion",
            "bell pepper",
            "garlic",
            "sweet potato",
            "cauliflower"
        }
    },


    "Salad trái cây": {
        "ingredients": {
            "apple",
            "banana",
            "orange",
            "mango",
            "grape",
            "kiwi",
            "pear",
            "pineapple",
            "pomegranate",
            "watermelon"
        }
    },


    "Sinh tố trái cây": {
        "ingredients": {
            "banana",
            "mango",
            "pineapple",
            "kiwi",
            "orange"
        }
    },


    "Khoai tây cà rốt": {
        "ingredients": {
            "potato",
            "carrot",
            "onion",
            "garlic"
        }
    },


    "Cà chua xào hành": {
        "ingredients": {
            "tomato",
            "onion",
            "garlic"
        }
    },


    "Ngô trộn rau củ": {
        "ingredients": {
            "corn",
            "cucumber",
            "tomato",
            "onion",
            "bell pepper"
        }
    },


    "Rau bina xào tỏi": {
        "ingredients": {
            "spinach",
            "garlic",
            "onion"
        }
    }
}


# ============================================================
# GỢI Ý MÓN ĂN
# ============================================================

def recommend_recipes(
    detected_foods
):

    ingredients = {

        normalize_food_name(
            food["name"]
        )

        for food in detected_foods
    }


    recommendations = []


    for recipe_name, recipe_data in (
        RECIPES.items()
    ):

        matched = sorted(

            ingredients
            &
            recipe_data[
                "ingredients"
            ]
        )


        if len(matched) == 0:
            continue


        score = len(
            matched
        )


        total_required = len(
            recipe_data[
                "ingredients"
            ]
        )


        coverage = (
            score
            / total_required
        ) * 100


        recommendations.append({

            "Recipe":
                recipe_name,

            "Matched":
                matched,

            "Score":
                score,

            "Coverage":
                coverage
        })


    recommendations.sort(

        key=lambda x: (
            x["Score"],
            x["Coverage"]
        ),

        reverse=True
    )


    return recommendations[:5]


# ============================================================
# TẠO CONTEXT CHO CHATBOT
# ============================================================

def build_recognition_context(
    detected_foods,
    nutrition_df,
    totals,
    recommendations
):

    lines = []


    lines.append(
        "KẾT QUẢ PHÂN TÍCH ẢNH:"
    )


    for food in detected_foods:

        lines.append(

            f"- {food['name']}: "
            f"{food['quantity']}, "
            f"{food['estimated_weight_grams']:.0f} g"
        )


    lines.append("")


    lines.append(
        "TỔNG DINH DƯỠNG:"
    )


    lines.append(

        f"- Calories: "
        f"{totals['Energy kcal']:.2f} kcal"
    )


    lines.append(

        f"- Carbs: "
        f"{totals['Carbs']:.2f} g"
    )


    lines.append(

        f"- Protein: "
        f"{totals['Protein(g)']:.2f} g"
    )


    lines.append(

        f"- Fat: "
        f"{totals['Fat(g)']:.2f} g"
    )


    lines.append(

        f"- Fibre: "
        f"{totals['Fibre(g)']:.2f} g"
    )


    lines.append("")


    lines.append(
        "MÓN ĂN ĐƯỢC GỢI Ý:"
    )


    for recipe in recommendations:

        lines.append(

            f"- {recipe['Recipe']}: "
            + ", ".join(
                recipe["Matched"]
            )
        )


    return "\n".join(
        lines
    )


# ============================================================
# CHATBOT SYSTEM PROMPT
# ============================================================

def build_system_instruction(
    config
):

    assistant_name = config.get(
        "assistant_name",
        "FoodBot"
    )


    functions = ", ".join(

        config.get(
            "functions",
            []
        )
    )


    out_of_scope = config.get(

        "out_of_scope_message",

        "Xin lỗi, mình chỉ hỗ trợ về thực phẩm."
    )


    return "\n".join([

        f"Bạn là {assistant_name}, "
        "trợ lý AI của dự án AI Food Recognition.",

        "Bạn hỗ trợ người dùng về "
        "rau củ quả, dinh dưỡng, món ăn "
        "và kết quả phân tích ảnh.",

        f"Các chức năng: {functions}.",

        "",

        "QUY TẮC:",

        "1. Trả lời bằng tiếng Việt.",

        "2. Thân thiện và dễ hiểu.",

        "3. Nếu người dùng hỏi về ảnh hiện tại, "
        "sử dụng kết quả phân tích ảnh.",

        "4. Không tự bịa số liệu dinh dưỡng.",

        "5. Ưu tiên số liệu từ food.csv.",

        "6. Khi gợi ý món ăn, ưu tiên "
        "những nguyên liệu người dùng đang có.",

        f"7. Nếu câu hỏi ngoài phạm vi: "
        f"{out_of_scope}"
    ])


# ============================================================
# CHAT HISTORY
# ============================================================

def build_history_text(
    messages,
    max_messages=8
):

    recent = (
        messages[-max_messages:]
    )


    lines = []


    for message in recent:

        role = (

            "Người dùng"

            if message["role"]
            == "user"

            else

            "FoodBot"
        )


        lines.append(

            f"{role}: "
            f"{message['content']}"
        )


    return "\n".join(
        lines
    )


# ============================================================
# ASK BOT
# ============================================================

def ask_bot(
    prompt,
    messages,
    client,
    food_df,
    config,
    recognition_context
):

    if client is None:

        return (
            "Gemini API chưa được kết nối. "
            "Hãy kiểm tra GEMINI_API_KEY."
        )


    system_instruction = (
        build_system_instruction(
            config
        )
    )


    history_text = (
        build_history_text(
            messages
        )
    )


    if recognition_context:

        recognition_text = (
            "\n\n"
            + recognition_context
        )

    else:

        recognition_text = (
            "\n\nChưa có ảnh được phân tích."
        )


    user_content = (

        "DỮ LIỆU DINH DƯỠNG TRONG FOOD.CSV:\n"

        + "\n".join(

            f"- {row['Food Items']}: "
            f"{row['Energy kcal']} kcal/100g, "
            f"Protein {row['Protein(g)']}g, "
            f"Carbs {row['Carbs']}, "
            f"Fat {row['Fat(g)']}g"

            for _, row in food_df.iterrows()
        )

        + recognition_text

        + "\n\nLỊCH SỬ:\n"

        + history_text

        + "\n\nCÂU HỎI:\n"

        + prompt
    )


    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=user_content,

            config=types.GenerateContentConfig(

                system_instruction=
                    system_instruction,

                temperature=0.4
            )
        )


        return (

            response.text

            or

            "Xin lỗi, mình chưa có câu trả lời."
        )


    except Exception as e:

        return (
            "Gemini API đang gặp lỗi.\n\n"
            f"Chi tiết: {e}"
        )


# ============================================================
# STREAMLIT APP
# ============================================================

def main():

    st.set_page_config(

        page_title=
            "AI Food Assistant",

        page_icon=
            "🥗",

        layout=
            "wide"
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    try:

        config = load_config()

        food_df = load_food_csv()

    except Exception as e:

        st.error(
            f"Lỗi khi đọc dữ liệu: {e}"
        )

        st.stop()


    # ========================================================
    # GEMINI
    # ========================================================

    api_key = load_api_key()

    client = create_client(
        api_key
    )


    # ========================================================
    # SESSION
    # ========================================================

    if "conversation_log" not in (
        st.session_state
    ):

        st.session_state.conversation_log = [

            {
                "role":
                    "assistant",

                "content":
                    config.get(

                        "initial_bot_message",

                        "Chào bạn! Mình là FoodBot 🥗."
                    )
            }
        ]


    if "recognition_context" not in (
        st.session_state
    ):

        st.session_state.recognition_context = ""


    # ========================================================
    # HEADER
    # ========================================================

    st.title(
        "🥗 AI Food Assistant"
    )


    st.write(
        "Tải ảnh rau củ quả lên để AI nhận diện, "
        "phân tích dinh dưỡng và gợi ý món ăn."
    )


    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.header(
            "⚙️ Trạng thái"
        )


        st.write(
            "Gemini SDK:",
            (
                "🟢 Sẵn sàng"
                if genai is not None
                else
                "🔴 Chưa cài"
            )
        )


        st.write(
            "Gemini API:",
            (
                "🟢 Đã kết nối"
                if client is not None
                else
                "🔴 Chưa có API key"
            )
        )


        st.markdown("---")


        st.subheader(
            "💡 Có thể hỏi"
        )


        st.markdown(
            "- Ảnh này có những gì?"
        )


        st.markdown(
            "- Tổng calories là bao nhiêu?"
        )


        st.markdown(
            "- Có món nào làm được không?"
        )


        st.markdown(
            "- Loại nào nhiều protein?"
        )


    # ========================================================
    # UPLOAD
    # ========================================================

    st.subheader(
        "📷 1. Tải ảnh rau củ quả"
    )


    uploaded_image = st.file_uploader(

        "Chọn ảnh",

        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )


    if uploaded_image is not None:

        image = Image.open(
            uploaded_image
        ).convert("RGB")


        st.image(

            image,

            caption=
                "Ảnh đã tải lên",

            use_container_width=True
        )


        if st.button(

            "🔍 Phân tích ảnh",

            type=
                "primary"
        ):


            if client is None:

                st.error(
                    "Chưa kết nối Gemini API."
                )


            else:

                with st.spinner(
                    "AI đang phân tích ảnh..."
                ):

                    try:

                        # ----------------------------------------
                        # Gemini
                        # ----------------------------------------

                        detected_foods = (
                            analyze_image(
                                client,
                                image
                            )
                        )


                        # ----------------------------------------
                        # Nutrition
                        # ----------------------------------------

                        nutrition_df = (
                            calculate_nutrition(
                                detected_foods,
                                food_df
                            )
                        )


                        totals = (
                            calculate_total_nutrition(
                                nutrition_df
                            )
                        )


                        # ----------------------------------------
                        # Recipes
                        # ----------------------------------------

                        recommendations = (
                            recommend_recipes(
                                detected_foods
                            )
                        )


                        # ----------------------------------------
                        # Context
                        # ----------------------------------------

                        st.session_state.recognition_context = (

                            build_recognition_context(

                                detected_foods,

                                nutrition_df,

                                totals,

                                recommendations
                            )
                        )


                        # ----------------------------------------
                        # Result
                        # ----------------------------------------

                        st.success(
                            "Đã phân tích ảnh thành công!"
                        )


                        st.subheader(
                            "🔎 2. Thực phẩm được nhận diện"
                        )


                        for food in detected_foods:

                            display_name = (
                                display_food_name(
                                    food["name"]
                                )
                            )

                            st.write(

                                f"**{display_name}** — "
                                f"{food['quantity']} — "
                                f"{food['estimated_weight_grams']:.0f} g"
                            )


                        # ----------------------------------------
                        # Nutrition
                        # ----------------------------------------

                        st.subheader(
                            "🥗 3. Dinh dưỡng"
                        )


                        # Tạo bảng hiển thị
                        # với tên tiếng Việt trong ngoặc

                        display_nutrition_df = (
                            nutrition_df.copy()
                        )


                        display_nutrition_df["Food"] = (
                            display_nutrition_df["Food"]
                            .apply(
                                display_food_name
                            )
                        )


                        st.dataframe(

                            display_nutrition_df,

                            use_container_width=
                                True,

                            hide_index=
                                True
                        )


                        # ----------------------------------------
                        # Total
                        # ----------------------------------------

                        st.subheader(
                            "🔥 Tổng dinh dưỡng"
                        )


                        col1, col2, col3, col4 = (
                            st.columns(4)
                        )


                        with col1:

                            st.metric(

                                "Calories",

                                f"{totals['Energy kcal']:.1f} kcal"
                            )


                        with col2:

                            st.metric(

                                "Protein",

                                f"{totals['Protein(g)']:.1f} g"
                            )


                        with col3:

                            st.metric(

                                "Carbs",

                                f"{totals['Carbs']:.1f} g"
                            )


                        with col4:

                            st.metric(

                                "Fat",

                                f"{totals['Fat(g)']:.1f} g"
                            )


                        # ----------------------------------------
                        # Recipes
                        # ----------------------------------------

                        st.subheader(
                            "🍽️ 4. Gợi ý món ăn"
                        )


                        if recommendations:

                            for i, recipe in enumerate(

                                recommendations,

                                1
                            ):

                                st.write(

                                    f"**{i}. "
                                    f"{recipe['Recipe']}**"
                                )


                                matched_display = [

                                    display_food_name(
                                        food
                                    )

                                    for food
                                    in recipe["Matched"]
                                ]


                                st.write(

                                    "Nguyên liệu phù hợp: "
                                    + ", ".join(
                                        matched_display
                                    )
                                )


                                st.write(

                                    f"Độ phù hợp: "
                                    f"{recipe['Coverage']:.1f}%"
                                )


                        else:

                            st.info(
                                "Chưa tìm thấy món phù hợp."
                            )


                    except Exception as e:

                        st.error(
                            f"Lỗi khi phân tích ảnh:\n\n{e}"
                        )


    # ========================================================
    # CHATBOT
    # ========================================================

    st.markdown("---")


    st.subheader(
        "💬 5. Chat với FoodBot"
    )


    for message in (
        st.session_state.conversation_log
    ):

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    prompt = st.chat_input(

        "Hỏi về ảnh, dinh dưỡng hoặc món ăn..."
    )


    if prompt:


        st.session_state.conversation_log.append({

            "role":
                "user",

            "content":
                prompt
        })


        with st.chat_message(
            "user"
        ):

            st.write(
                prompt
            )


        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "FoodBot đang trả lời..."
            ):

                reply = ask_bot(

                    prompt=

                        prompt,

                    messages=

                        st.session_state.conversation_log,

                    client=

                        client,

                    food_df=

                        food_df,

                    config=

                        config,

                    recognition_context=

                        st.session_state.recognition_context
                )


                st.write(
                    reply
                )


        st.session_state.conversation_log.append({

            "role":
                "assistant",

            "content":
                reply
        })


    # ========================================================
    # CONTEXT
    # ========================================================

    with st.expander(
        "🔎 Kết quả đang được chatbot sử dụng"
    ):

        if st.session_state.recognition_context:

            st.text(
                st.session_state.recognition_context
            )

        else:

            st.info(
                "Chưa có ảnh được phân tích."
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()