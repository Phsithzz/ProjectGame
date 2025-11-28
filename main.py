import streamlit as st
st.set_page_config(layout="wide", page_title="Game Battle")
with st.container(border=True):

    st.markdown("<h1 style='text-align:center;'>Game Battle</h1>",unsafe_allow_html=True)
    col_enemy, col_player = st.columns(2)
    with col_enemy:
         
        with st.container(border=True):

                st.markdown("<h1 style='text-align:center;'>Enemy</h1>",unsafe_allow_html=True)
                st.image("images/enemy.jpg")
                st.markdown("""
    <div style='margin-bottom:20px; gap:5px; background-color:red; width:100%; height:60px; display:flex; align-items:center; padding:2px; box-sizing:border-box;'>
        <p style='margin:0; color:white; font-weight: 600;'>Enemy HP:</p>  
        <div style='background-color:#08CB00; width:70%; height:100%;'></div>
    </div>
    """, unsafe_allow_html=True) 
                
    with col_player:
         
        with st.container(border =True):

            st.markdown("<h1 style='text-align:center;'>Camera Player </h1>",unsafe_allow_html=True)
            st.image("images/player.jpg")
            st.markdown("""
    <div style='margin-bottom:20px;gap:5px;background-color:red; width:100%; height:60px; display:flex; align-items:center; padding:2px; box-sizing:border-box;'>
        <p style='margin:0;color:white;font-weight: 600;'>Player HP:</p>  
        <div style='background-color:#08CB00; width:80%; height:100%;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container(border= True):
         
         st.markdown("ท่าที่ต้องทำ: Push-up 10 reps")
         st.markdown("Timer: 15 s")
         st.markdown("คะแนน: 85 (Good)")

    
    with st.container(border=True):
         punch_col, guard_col, kick_col = st.columns(3)

         with punch_col:
              st.button("Punch ", use_container_width=True) 
         with guard_col:
              st.button("Guard ", use_container_width=True)
         with kick_col:
              st.button("Kick ", use_container_width=True)




