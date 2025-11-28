import streamlit as st

#อันนี้ลองใช้tailwind และเขียนแบบเว็บเฉยๆ เผื่อได้ใช้
#อันนี้ลองใช้tailwind และเขียนแบบเว็บเฉยๆ เผื่อได้ใช้
#อันนี้ลองใช้tailwind และเขียนแบบเว็บเฉยๆ เผื่อได้ใช้

html_code = f"""
<script src="https://cdn.tailwindcss.com"></script>
<div class="w-[800px] mx-auto border-2 border-black rounded-xl p-5 bg-gray-900 text-white">
    <h1 class="text-center text-3xl font-bold mb-4">Game Battle</h1>

<div class="border-2 border-red-500 p-4 rounded-xl text-center mb-4 bg-gray-800">
        <p class="text-xl font-bold text-red-400">Enemy HP: 100</p>
<div class="w-32 h-32 bg-red-900 mx-auto mt-2 rounded-full flex items-center justify-center">Enemy</div>
</div>

<div class="grid grid-cols-2 gap-4 mb-4">
<div class="border-2 border-blue-500 p-2 rounded-xl text-center bg-gray-800 flex flex-col items-center justify-center" style="min-height: 300px;">
            <h2 class="text-xl font-bold text-blue-400 mb-2">Camera Feed</h2>

</div>
<div class="border-2 border-green-500 p-4 rounded-xl bg-gray-800 flex flex-col justify-center font-bold text-lg">
            <p>ท่าที่ต้องทำ: <span class="text-yellow-400">Push-up</span></p>
            <p>ทำไปแล้ว: <span class="text-green-400 text-2xl">Push up</span> reps</p>
            <p>คะแนนรวม: <span class="text-blue-400">85</span></p>
</div>
</div>
<div class="border-2 border-orange-500 p-4 rounded-xl text-center">
        Player HP: 100
</div>
</div>
"""

st.markdown(html_code, unsafe_allow_html=True)
