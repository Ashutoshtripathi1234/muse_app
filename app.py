import streamlit as st
import os
import yaml
import shutil
from omegaconf import OmegaConf
import tempfile
import subprocess

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as f:
        f.write(uploaded_file.getvalue())
        return f.name

def create_inference_config(video_path, audio_path):
    config = {
        "task1": {
            "video_path": video_path,
            "audio_path": audio_path,
            "bbox_shift": 0
        }
    }
    config_path = os.path.abspath("inference_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

def run_musetalk(video_path, audio_path, use_float16=False, batch_size=8):
    config_path = create_inference_config(video_path, audio_path)
    result_dir = os.path.abspath("results")
    os.makedirs(result_dir, exist_ok=True)
    
    cmd = [
        "python", "main.py",
        "--inference_config", config_path,
        "--result_dir", result_dir,
        "--batch_size", str(batch_size)
    ]
    
    if use_float16:
        cmd.append("--use_float16")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return process, result_dir, config_path

def main():
    st.title("MuseTalk Interface")
    
    video_file = st.file_uploader("Upload Video/Image", type=['mp4', 'png', 'jpg', 'jpeg'])
    audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
    
    col1, col2 = st.columns(2)
    with col1:
        use_float16 = st.checkbox("Use Float16 (Faster)")
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, value=8)
    
    if st.button("Process", disabled=not (video_file and audio_file)):
        with st.spinner("Processing..."):
            video_path = save_uploaded_file(video_file)
            audio_path = save_uploaded_file(audio_file)
            
            process, result_dir, config_path = run_musetalk(video_path, audio_path, use_float16, batch_size)
            
            progress_bar = st.progress(0)
            output_text = st.empty()
            error_text = st.empty()
            
            output = []
            for stdout_line in process.stdout:
                output.append(stdout_line.strip())
                output_text.text("\n".join(output[-5:]))
                progress = min(len(output) / 100, 1.0)
                progress_bar.progress(progress)
            
            stderr = process.stderr.read()
            if stderr and "No such file or directory" not in stderr:
                error_text.error(f"Error output:\n{stderr}")
            
            process.wait()
            
            result_video = os.path.join(result_dir, f"{os.path.splitext(video_file.name)[0]}_{os.path.splitext(audio_file.name)[0]}.mp4")
            
            if os.path.exists(result_video):
                st.success("Processing complete!")
                st.video(result_video)
                
                with open(result_video, 'rb') as f:
                    st.download_button(
                        label="Download Video",
                        data=f,
                        file_name=os.path.basename(result_video),
                        mime='video/mp4'
                    )
            else:
                st.error("Error: Output video not generated")
            
            os.unlink(video_path)
            os.unlink(audio_path)
            os.unlink(config_path)

if __name__ == "__main__":
    main()