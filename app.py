import streamlit as st
import os
import cv2
import numpy as np
from ransac import *
from match_features import *
from scipy import optimize
from optimize_fcn import *
from PIL import Image
import tempfile

class GenerateMosaic:

    def __init__(self, image_files):
        self.img_all = {}
        self.image_files = image_files  # List of uploaded file objects
        self.middle_id = int(np.floor(len(image_files) / 2))
        # Create a temporary directory to store uploaded images
        self.temp_dir = tempfile.mkdtemp()
        self.img_paths = self._save_uploaded_images()
        
    def _save_uploaded_images(self):
        """Save uploaded images to temp directory and return their paths"""
        img_paths = []
        for i, img_file in enumerate(self.image_files):
            img_path = os.path.join(self.temp_dir, f"image_{i}.jpg")
            with open(img_path, "wb") as f:
                f.write(img_file.getbuffer())
            img_paths.append(img_path)
        return img_paths

    def mosaic(self):
        H_all = {}
        for i in range(len(self.img_paths) - 1):
            st.write(f"Processing image {i+1} & image {i+2}")

            key = f'H{i}{i + 1}'

            img_1_path = self.img_paths[i]
            img_2_path = self.img_paths[i + 1]
            
            col1, gap, col2, a, b, c, v, f, d = st.columns(9)

            with col1:
                st.image(img_1_path, width=200)
                
            with col2:
                st.image(img_2_path, width=200)
                
            # Get SIFT descriptors
            siftmatch_obj = SiftMatching(img_1_path, img_2_path, results_fldr=self.temp_dir, nfeatures=2000, gamma=0.6)
            correspondence = siftmatch_obj.run()

            # Run RANSAC to remove outliers
            ransac_obj = RANSAC()
            inliers_cnt, inliers, outliers, sample_pts, final_H = ransac_obj.run_ransac(correspondence)

            # Draw inliers and outliers
            result_path = os.path.join(siftmatch_obj.result_fldr, f'{siftmatch_obj.prefix}_inliers.jpg')
            ransac_obj.draw_lines(np.concatenate((inliers, sample_pts), axis=0), siftmatch_obj.img_1_bgr,
                                  siftmatch_obj.img_2_bgr, result_path,
                                  line_color=RANSAC._GREEN, pt_color=[0, 0, 0])

            result_path = os.path.join(siftmatch_obj.result_fldr, f'{siftmatch_obj.prefix}_outliers.jpg')
            ransac_obj.draw_lines(outliers, siftmatch_obj.img_1_bgr, siftmatch_obj.img_2_bgr, result_path,
                                  line_color=RANSAC._RED, pt_color=[0, 0, 0])

            # Optimize the homography
            x = np.concatenate((inliers, sample_pts), axis=0)
            opt_obj = OptimizeFunction(fun=fun_LM_homography, x0=final_H.flatten(), jac=jac_LM_homography,
                                       args=(x[:, 0:2], x[:, 2:]))
            LM_sol = opt_obj.levenberg_marquardt(delta_thresh=1e-24, tau=0.8)

            H_all[key] = LM_sol.x.reshape(3, 3)
            H_all[key] = H_all[key] / H_all[key][-1, -1]

        H_all = self.compute_H_wrt_middle_img(H_all)
        self.stitch(H_all, siftmatch_obj.result_fldr)

    def stitch(self, H_all, result_fldr):
        canvas_img, mask, offset = self.get_blank_canvas(H_all)

        for i, img_path in enumerate(self.img_paths):
            key = f"H{i}{self.middle_id}"
            H = H_all[key]

            img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            canvas_img = fit_image_in_target_space(img_rgb, canvas_img, mask, np.linalg.inv(H),
                                                   offset=offset)  
            mask[np.where(canvas_img)[0:2]] = 0

            result_path = os.path.join(result_fldr, f'panorama_{i}.jpg')
            cv2.imwrite(result_path, canvas_img[:, :, (2, 1, 0)])

        # Show the final panorama
        final_img_path = os.path.join(result_fldr, f'panorama_{len(self.img_paths)-1}.jpg')
        if os.path.exists(final_img_path):
            img = Image.open(final_img_path)
            st.image(img, caption="Final Panorama", use_column_width=True)
            
            # Add download button for the final image
            with open(final_img_path, "rb") as file:
                btn = st.download_button(
                    label="Download Panorama",
                    data=file,
                    file_name="panorama.jpg",
                    mime="image/jpeg"
                )

    def get_blank_canvas(self, H_all):
        img_path = self.img_paths[0]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        img_h, img_w, _ = img.shape

        min_crd_canvas = np.array([np.inf, np.inf, np.inf])
        max_crd_canvas = np.array([-np.inf, -np.inf, -np.inf])

        for i in range(len(self.img_paths)):
            key = f"H{i}{self.middle_id}"
            H = H_all[key]
            min_crd, max_crd = self.compute_extent(H, img_w, img_h)

            min_crd_canvas = np.minimum(min_crd, min_crd_canvas)
            max_crd_canvas = np.maximum(max_crd, max_crd_canvas)

        width_canvas = np.ceil(max_crd_canvas - min_crd_canvas)[0] + 1
        height_canvas = np.ceil(max_crd_canvas - min_crd_canvas)[1] + 1

        canvas_img = np.zeros((int(height_canvas), int(width_canvas), 3), dtype=np.int64)

        offset = min_crd_canvas.astype(np.int64)
        offset[2] = 0  

        mask = np.ones((int(height_canvas), int(width_canvas)))

        return canvas_img, mask, offset

    def compute_extent(self, H, img_w, img_h):
        corners_img = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])

        t_one = np.ones((corners_img.shape[0], 1))
        t_out_pts = np.concatenate((corners_img, t_one), axis=1)
        canvas_crd_corners = np.matmul(H, t_out_pts.T)
        canvas_crd_corners = canvas_crd_corners / canvas_crd_corners[-1, :]  

        min_crd = np.amin(canvas_crd_corners.T, axis=0)  
        max_crd = np.amax(canvas_crd_corners.T, axis=0)
        return min_crd, max_crd

    def compute_H_wrt_middle_img(self, H_all):
        num_imgs = len(H_all) + 1
        key = f"H{self.middle_id}{self.middle_id}"
        H_all[key] = np.eye(3)

        for i in range(0, self.middle_id):
            key = f"H{i}{self.middle_id}"
            j = i
            temp = np.eye(3)
            while j < self.middle_id:
                key_t = f"H{j}{j + 1}"
                temp = np.matmul(H_all[key_t], temp)
                j += 1

            H_all[key] = temp

        for i in range(self.middle_id + 1, num_imgs):
            key = f"H{i}{self.middle_id}"

            temp = np.eye(3)
            j = i - 1

            while j >= self.middle_id:
                key_t = f"H{j}{j + 1}"
                temp = np.matmul(np.linalg.inv(H_all[key_t]), temp)
                j -= 1

            H_all[key] = temp

        return H_all

# Streamlit Interface
st.title("Mosaic Generator")

# User Input: Direct Image Upload
uploaded_files = st.file_uploader("Upload images for mosaic generation", 
                                  type=["jpg", "jpeg", "png"], 
                                  accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) < 2:
        st.warning("Please upload at least two images for mosaic generation.")
    else:
        st.success(f"{len(uploaded_files)} images uploaded!")
        
        # Display thumbnails of uploaded images
        cols = st.columns(min(5, len(uploaded_files)))
        for i, img_file in enumerate(uploaded_files):
            cols[i % len(cols)].image(img_file, caption=f"Image {i+1}", width=150)
        
        if st.button("Generate Mosaic"):
            with st.spinner("Processing images and generating mosaic..."):
                obj = GenerateMosaic(image_files=uploaded_files)
                obj.mosaic()
else:
    st.info("Please upload at least two images to generate a mosaic.")
    
st.markdown("""
### Tips for best results:
1. Upload images in the order they should be stitched (left to right or right to left)
2. For best results, use images taken from the same position with rotation only
3. Higher resolution images will produce better mosaics but take longer to process
""")