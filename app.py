import streamlit as st
from backend import record_audio, extract_embedding, save_voiceprint, verify_user, recognize_user, detect_emotion

st.title("Voice Authentication with ECAPA-TDNN")

mode = st.sidebar.radio("Choose an option", ["Enroll User", "Speaker Verification", "Speaker Recognition", "Emotion Detection"])

if mode == "Enroll User":
    name = st.text_input("Enter user ID:")
    if st.button("Record & Enroll"):
        if not name:
            st.warning("Please enter a user ID before enrolling.")
        else:
            try:
                record_audio("input.wav")
                emb = extract_embedding("input.wav")
                save_voiceprint(name, emb)
                st.success(f"User '{name}' enrolled successfully!")
            except Exception as e:
                st.error(f"Enrollment failed: {e}")

elif mode == "Speaker Verification":
    claimed_name = st.text_input("Enter claimed user ID:")
    if st.button("Record & Verify"):
        if not claimed_name:
            st.warning("Please enter a user ID to verify.")
        else:
            try:
                record_audio("input.wav")
                emb = extract_embedding("input.wav")
                result, msg = verify_user(claimed_name, emb)
                if result:
                    st.success(f"Verified: {msg}")
                else:
                    st.error(f"Verification Failed: {msg}")
            except Exception as e:
                st.error(f"Verification failed: {e}")

elif mode == "Speaker Recognition":
    if st.button("Record & Recognize"):
        try:
            record_audio("input.wav")
            emb = extract_embedding("input.wav")
            name, score = recognize_user(emb)
            if name:
                st.success(f"Recognized as: {name} (Similarity: {score:.2f})")
            else:
                st.error("No match found.")
        except Exception as e:
            st.error(f"Recognition failed: {e}")

elif mode == "Emotion Detection":
    if st.button("Record & Detect Emotion"):
        try:
            record_audio("input.wav")
            emotion = detect_emotion("input.wav")
            st.success(f"Detected Emotion: {emotion}")
        except Exception as e:
            st.error(f"Emotion detection failed: {e}")
