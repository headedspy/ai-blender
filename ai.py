import bpy
import math
import bgl
import random
import cv2
import os
import uuid
from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty,
                       )       
from bpy.types import (Panel,
                       Menu,
                       Operator,
                       PropertyGroup,
                       )
import numpy as np
from tensorflow import keras
import sklearn

from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

def rgb_to_hsv(arr):
    arr = np.asarray(arr)

    # check length of the last dimension, should be _some_ sort of rgb
    if arr.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {} was found.".format(arr.shape))

    in_shape = arr.shape
    arr = np.array(
        arr, copy=False,
        dtype=np.promote_types(arr.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )
    out = np.zeros_like(arr)
    arr_max = arr.max(-1)
    ipos = arr_max > 0
    delta = arr.ptp(-1)
    s = np.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # red is max
    idx = (arr[..., 0] == arr_max) & ipos
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
    # green is max
    idx = (arr[..., 1] == arr_max) & ipos
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
    # blue is max
    idx = (arr[..., 2] == arr_max) & ipos
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

    out[..., 0] = (out[..., 0] / 6.0) % 1.0
    out[..., 1] = s
    out[..., 2] = arr_max

    return out.reshape(in_shape)

# Properties storage
class Properties(PropertyGroup):
    ground: BoolProperty(
        name="",
        description="Generate a ground plane",
        default = False
    )
    
    nrObj: IntProperty(
        name = "",
        description="Number of objects",
        default = 1,
        min = 1,
        max = 6
    )
    
    groundBias: FloatProperty(
        name = "",
        description = "Ground Bias",
        default = 0,
        min = 0,
        max = 1
    )
    
    rndGroundBias: BoolProperty(
        name="",
        description="Randomize the Ground Bias",
        default = False
    )
    
    ratings: EnumProperty(
        items=[
            ('1', '1', 'Terrible', '', 0),
            ('2', '2', 'Passable', '', 1),
            ('3', '3', 'Amazing', '', 2)
        ],
        default='3'
    )
    
    filepath_generate: StringProperty(
        name = "Path",
        description = "Path to the folder containing the files to import",
        default = "",
        subtype = 'DIR_PATH'
    )
    
    filepath_train: StringProperty(
        name = "Path",
        description = "Path to the folder containing the labelled images for training",
        default = "",
        subtype = 'DIR_PATH'
    )
    
    info_noGround: StringProperty(
        name =  "info_noground",
        default = "0 0 0 0 0 0"
    )
    
    info_Ground: StringProperty(
        name =  "info_ground",
        default = "0 0 0 0 0 0"
    )
    
    info_n1: StringProperty(
        name = "info_n1",
        default = "1| 0   0   0   0   0   0"
    )
    info_n2: StringProperty(
        name = "info_n2",
        default = "2| 0   0   0   0   0   0"
    )
    info_n3: StringProperty(
        name = "info_n3",
        default = "3| 0   0   0   0   0   0"
    )
    info_g1: StringProperty(
        name = "info_g1",
        default = "1| 0   0   0   0   0   0"
    )
    info_g2: StringProperty(
        name = "info_g2",
        default = "2| 0   0   0   0   0   0"
    )
    info_g3: StringProperty(
        name = "info_g3",
        default = "3| 0   0   0   0   0   0"
    )
        
    filepath_export: StringProperty(
        name = "Path",
        description = "Path to the folder to export the trained model",
        default = "",
        subtype = 'DIR_PATH'
    )
    
    epochs: IntProperty(
        name = "",
        description="Epochs to Train for",
        default = 20,
        min = 1,
        max = 100
    )
    
    filepath_import: StringProperty(
        name = "Path",
        description = "Path to the trained model",
        default = "",
        subtype = 'FILE_PATH'
    )
    
    comp_rating: IntProperty(
        name = "",
        default = 1
    )
    
    rating_bar1: StringProperty(
        name = "Rating 1",
        default = "1 ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒"
    )
    
    rating_bar2: StringProperty(
        name = "Rating 2",
        default = "2 ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒"
    )
    
    rating_bar3: StringProperty(
        name = "Rating 3",
        default = "3 ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒"
    )
    
    color_1: StringProperty(
        name = "Color 1",
        default = ""
    )
    color_2: StringProperty(
        name = "Color 2",
        default = ""
    )
    color_3: StringProperty(
        name = "Color 3",
        default = ""
    )
    color_4: StringProperty(
        name = "Color 4",
        default = ""
    )
    color_5: StringProperty(
        name = "Color 5",
        default = ""
    )
    color_6: StringProperty(
        name = "Color 6",
        default = ""
    )
    color_7: StringProperty(
        name = "Color 7",
        default = ""
    )
    color_8: StringProperty(
        name = "Color 8",
        default = ""
    )
    color_9: StringProperty(
        name = "Color 9",
        default = ""
    )
    color_10: StringProperty(
        name = "Color 10",
        default = ""
    )
    color_11: StringProperty(
        name = "Color 11",
        default = ""
    )
    color_12: StringProperty(
        name = "Color 12",
        default = ""
    )

class Constants():
    #minimal distance from camera
    min_z = 50
    
    #max object scale on an axis
    max_scale = 5

# output messagebox function
def ShowMessageBox(message = "", title = "Message Box", icon = 'INFO'):
    def draw(self, context):
        self.layout.label(text=message)

    bpy.context.window_manager.popup_menu(draw, title = title, icon = icon)

# Create the scene
class GenerateScene(bpy.types.Operator):
    bl_idname = "wm.generate_scene"
    bl_label = "Scene Generation"
    
    def execute(self, context):
        scene = context.scene
        properties = scene.custom_properties
        
        # clear the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # create camera
        rnd = random.random()
        pos_z = rnd * 20.0
        rot_x = rnd 
        
        camera = bpy.ops.object.camera_add(
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, pos_z),
            rotation=(math.radians(90.0), 0, 0)
        )
        
        camera = context.object
        camera.scale = (100,100,100)
        
        camera.data.show_composition_thirds = True
        camera.data.show_composition_center = True
        camera.data.show_composition_center_diagonal = True
        
        #calculate camera frustum
        frustum = []
        matrix = camera.matrix_basis
        frame = [matrix @ v for v in camera.data.view_frame(scene=scene)]
        origin = matrix.to_translation()
        frame.append(origin)
        
        for p in frame:
            bpy.ops.object.empty_add(location=p)
            frustum.append(context.object)
        
        camera.data.lens = 32
        
        # create ground plane
        if properties.ground:
            bpy.ops.mesh.primitive_plane_add(
                enter_editmode=False,
                align='WORLD',
                location=(0, 50, 0),
                size=200
            )

        # generate nr of primitives
        for i in range(properties.nrObj):
            obj_type = int(random.random() * 7)
            
            if obj_type == 0:
                bpy.ops.mesh.primitive_cube_add()
            if obj_type == 1:
                bpy.ops.mesh.primitive_uv_sphere_add()
            if obj_type == 2:
                bpy.ops.mesh.primitive_ico_sphere_add()
            if obj_type == 3:
                bpy.ops.mesh.primitive_cylinder_add()
            if obj_type == 4:
                bpy.ops.mesh.primitive_cone_add()
            if obj_type == 5:
                bpy.ops.mesh.primitive_torus_add()
            if obj_type == 6:
                bpy.ops.mesh.primitive_monkey_add()

            object = context.object
            
            #calculate a random point within the camera frustum
            scene_size_x = (frustum[1].location.x * 2)
            scene_size_y = (frustum[0].location.z + (frustum[1].location.z * -1))
            scene_size_z = frustum[0].location.y
            
            pos_z = random.random() * scene_size_z + Constants.min_z
            if pos_z > scene_size_z:
                pos_z = scene_size_z
            
            calc_x = (pos_z * (scene_size_x/2)) / (scene_size_z)
            pos_x = random.random() * calc_x * (-1 if random.random() >= 0.5 else 1)
            
            # include ground bias
            if properties.rndGroundBias:
                properties.groundBias = random.random()
            
            if properties.ground and random.random() < properties.groundBias:
                pos_y = 0
            else:
                calc_y = (pos_z * (scene_size_y/2)) / (scene_size_z)
                pos_y = random.random() * calc_y
            
            object.location = (pos_x, pos_z, pos_y)
            
            object.rotation_euler = (math.radians(random.randint(1, 8)*45), 
                                     math.radians(random.randint(1, 8)*45), 
                                     math.radians(random.randint(1, 8)*45))
            
            object.scale = (random.random() * Constants.max_scale + 1, 
                            random.random() * Constants.max_scale + 1, 
                            random.random() * Constants.max_scale + 1)

        # switch to camera perspective
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].region_3d.view_perspective = 'CAMERA'
                break
        
        # deselect all
        bpy.ops.object.select_all(action='TOGGLE')

        return {'FINISHED'}

class SaveScene(bpy.types.Operator):
    bl_idname = "wm.save_scene"
    bl_label = "OPOPOP"
    
    def execute(self, context):
        
        # turn off overlays
        bpy.context.space_data.overlay.show_overlays = False
        
        scene = context.scene
        properties = scene.custom_properties
        
        # check if filepath is set
        if not properties.filepath_generate:
            ShowMessageBox("Error: Filepath field is empty", "No Filepath Set", 'ERROR')
            return{'CANCELLED'}
        
        # render viewport and render image
        sce = bpy.context.scene.name
        bpy.ops.render.opengl(write_still=True)
        bpy.data.images["Render Result"].save_render("C:\image.png")
        
        # convert to grayscale
        image = cv2.imread("C:\image.png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #save grayscale
        name = (str(properties.ratings) + "_") + (("0"+str(properties.nrObj)) if properties.nrObj != 10 else (str(properties.nrObj))) + ("_G." if properties.ground else "_N.") + str(uuid.uuid4()) + ".png"
        path = bpy.path.abspath(properties.filepath_generate) + name
        
        cv2.imwrite(path, gray)
        
        # delete temp
        os.remove("C:/image.png")
        
        # turn on overlays
        bpy.context.space_data.overlay.show_overlays = True
        
        return {'FINISHED'}

class DataInfo(bpy.types.Operator):
    bl_idname = "wm.info"
    bl_label = "a"
    
    def execute(self, context):
        scene = context.scene
        properties = scene.custom_properties
        
        # check if filepath is set
        if not properties.filepath_train:
            ShowMessageBox("Error: Filepath field is empty", "No Filepath Set", 'ERROR')
            return{'CANCELLED'}
        
        #counters
        n1 = n2 = n3 = n4 = n5 = n6 = 0
        g1 = g2 = g3 = g4 = g5 = g6 = 0
        
        i_n1_1 = i_n2_1 = i_n3_1 = i_n4_1 = i_n5_1 = i_n6_1 = 0
        i_n1_2 = i_n2_2 = i_n3_2 = i_n4_2 = i_n5_2 = i_n6_2 = 0
        i_n1_3 = i_n2_3 = i_n3_3 = i_n4_3 = i_n5_3 = i_n6_3 = 0
        
        i_g1_1 = i_g2_1 = i_g3_1 = i_g4_1 = i_g5_1 = i_g6_1 = 0
        i_g1_2 = i_g2_2 = i_g3_2 = i_g4_2 = i_g5_2 = i_g6_2 = 0
        i_g1_3 = i_g2_3 = i_g3_3 = i_g4_3 = i_g5_3 = i_g6_3 = 0
        
        for filename in os.listdir(bpy.path.abspath(properties.filepath_train)):
            if filename[5] == 'G':
                if filename[3] == '1':
                    g1 = g1 + 1
                    if filename[0] == '1':
                        i_g1_1 = i_g1_1 + 1
                    if filename[0] == '2':
                        i_g1_2 = i_g1_2 + 1
                    if filename[0] == '3':
                        i_g1_3 = i_g1_3 + 1
                elif filename[3] == '2':
                    g2 = g2 + 1
                    if filename[0] == '1':
                        i_g2_1 = i_g2_1 + 1
                    if filename[0] == '2':
                        i_g2_2 = i_g2_2 + 1
                    if filename[0] == '3':
                        i_g2_3 = i_g2_3 + 1
                elif filename[3] == '3':
                    g3 = g3 + 1
                    if filename[0] == '1':
                        i_g3_1 = i_g3_1 + 1
                    if filename[0] == '2':
                        i_g3_2 = i_g3_2 + 1
                    if filename[0] == '3':
                        i_g3_3 = i_g3_3 + 1
                elif filename[3] == '4':
                    g4 = g4 + 1
                    if filename[0] == '1':
                        i_g4_1 = i_g4_1 + 1
                    if filename[0] == '2':
                        i_g4_2 = i_g4_2 + 1
                    if filename[0] == '3':
                        i_g4_3 = i_g4_3 + 1
                elif filename[3] == '5':
                    g5 = g5 + 1
                    if filename[0] == '1':
                        i_g5_1 = i_g5_1 + 1
                    if filename[0] == '2':
                        i_g5_2 = i_g5_2 + 1
                    if filename[0] == '3':
                        i_g5_3 = i_g5_3 + 1
                elif filename[3] == '6':
                    g6 = g6 + 1
                    if filename[0] == '1':
                        i_g6_1 = i_g6_1 + 1
                    if filename[0] == '2':
                        i_g6_2 = i_g6_2 + 1
                    if filename[0] == '3':
                        i_g6_3 = i_g6_3 + 1
            elif filename[5] == 'N':
                if filename[3] == '1':
                    n1 = n1 + 1
                    if filename[0] == '1':
                        i_n1_1 = i_n1_1 + 1
                    if filename[0] == '2':
                        i_n1_2 = i_n1_2 + 1
                    if filename[0] == '3':
                        i_n1_3 = i_n1_3 + 1
                elif filename[3] == '2':
                    n2 = n2 + 1
                    if filename[0] == '1':
                        i_n2_1 = i_n2_1 + 1
                    if filename[0] == '2':
                        i_n2_2 = i_n2_2 + 1
                    if filename[0] == '3':
                        i_n2_3 = i_n2_3 + 1
                elif filename[3] == '3':
                    n3 = n3 + 1
                    if filename[0] == '1':
                        i_n3_1 = i_n3_1 + 1
                    if filename[0] == '2':
                        i_n3_2 = i_n3_2 + 1
                    if filename[0] == '3':
                        i_n3_3 = i_n3_3 + 1
                elif filename[3] == '4':
                    n4 = n4 + 1
                    if filename[0] == '1':
                        i_n4_1 = i_n4_1 + 1
                    if filename[0] == '2':
                        i_n4_2 = i_n4_2 + 1
                    if filename[0] == '3':
                        i_n4_3 = i_n4_3 + 1
                elif filename[3] == '5':
                    n5 = n5 + 1
                    if filename[0] == '1':
                        i_n5_1 = i_n5_1 + 1
                    if filename[0] == '2':
                        i_n5_2 = i_n5_2 + 1
                    if filename[0] == '3':
                        i_n5_3 = i_n5_3 + 1
                elif filename[3] == '6':
                    n6 = n6 + 1
                    if filename[0] == '1':
                        i_n6_1 = i_n6_1 + 1
                    if filename[0] == '2':
                        i_n6_2 = i_n6_2 + 1
                    if filename[0] == '3':
                        i_n6_3 = i_n6_3 + 1
                
                properties.info_Ground = str(g1) + " " + str(g2) + " " + str(g3) + " " + str(g4) + " " + str(g5) + " " + str(g6)
                properties.info_noGround = str(n1) + " " + str(n2) + " " + str(n3) + " " + str(n4) + " " + str(n5) + " " + str(n6)
                
                properties.info_n1 = str(i_n1_1) + " " + str(i_n2_1) + " " + str(i_n3_1) + " " + str(i_n4_1) + " " + str(i_n5_1) + " " + str(i_n6_1)
                properties.info_n2 = str(i_n1_2) + " " + str(i_n2_2) + " " + str(i_n3_2) + " " + str(i_n4_2) + " " + str(i_n5_2) + " " + str(i_n6_2)
                properties.info_n3 = str(i_n1_3) + " " + str(i_n2_3) + " " + str(i_n3_3) + " " + str(i_n4_3) + " " + str(i_n5_3) + " " + str(i_n6_3)

                properties.info_g1 = str(i_g1_1) + " " + str(i_g2_1) + " " + str(i_g3_1) + " " + str(i_g4_1) + " " + str(i_g5_1) + " " + str(i_g6_1)
                properties.info_g2 = str(i_g1_2) + " " + str(i_g2_2) + " " + str(i_g3_2) + " " + str(i_g4_2) + " " + str(i_g5_2) + " " + str(i_g6_2)
                properties.info_g3 = str(i_g1_3) + " " + str(i_g2_3) + " " + str(i_g3_3) + " " + str(i_g4_3) + " " + str(i_g5_3) + " " + str(i_g6_3)

        return {'FINISHED'}

class TrainNN(bpy.types.Operator):
    bl_idname = "wm.train"
    bl_label = "a"
    
    def execute(self, context):
        scene = context.scene
        properties = scene.custom_properties
        
        # check if filepath is set
        if not properties.filepath_train:
            ShowMessageBox("Error: Filepath field is empty", "No Filepath Set", 'ERROR')
            return{'CANCELLED'}
        
        train_image = []
        labels = []
        
        for filename in os.listdir(bpy.path.abspath(properties.filepath_train)):
            img = keras.utils.load_img(os.path.join(bpy.path.abspath(properties.filepath_train), filename), target_size = (68, 120 , 3), grayscale = False)
            img = keras.utils.img_to_array(img)
            #[0-255] -> [0-1]
            img = img/255
            train_image.append(img)
            # append score
            score = int(filename[0])
            
            score = score - 1 # this is because To categorical works from 0 to number of classes

            labels.append(score)
            
            print(filename)
        
        X = np.array(train_image)
        y = np.array(labels)
        print (X.shape)
        print (y.shape)
        
        y_cat=keras.utils.to_categorical(y, num_classes = 3)
        print(y_cat.shape)
        
        model = keras.models.Sequential()
        
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(68,120,3)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(keras.layers.Dropout(0.45))
        model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(keras.layers.Dropout(0.45))
        model.add(keras.layers.Dense(3, activation='softmax'))
        
        model.summary()

        #model.compile(loss='mean_squared_error',optimizer='Adam',metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
        
        import sklearn.model_selection
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y_cat, random_state=42, test_size=0.2)
        
        #callback = [callbacks.EarlyStopping(monitor='accuracy', patience=5), LearningRateReducerCb()]
        callback = [callbacks.EarlyStopping(monitor='accuracy', patience=5)]
        
        model.fit(X_train, y_train, epochs=properties.epochs, validation_data=(X_test, y_test), callbacks=callback)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        
        print(test_acc)
        model.save(os.path.join(bpy.path.abspath(properties.filepath_export), str(test_acc)+'.h5'))
        
        keras.backend.clear_session()
        
        return {'FINISHED'}

class UseNN(bpy.types.Operator):
    bl_idname = "wm.use"
    bl_label = "a"
    
    def execute(self, context):
        scene = context.scene
        properties = scene.custom_properties
        
        model = keras.models.load_model(bpy.path.abspath(properties.filepath_import))
        
        probability_model = keras.Sequential([
          model,
          keras.layers.Softmax()
        ])
        
        # render viewport
        sce = bpy.context.scene.name
        
        bpy.context.space_data.overlay.show_overlays = False
        
        bpy.ops.render.opengl(write_still=True)
        bpy.data.images["Render Result"].save_render("C:\image.png")
        
        bpy.context.space_data.overlay.show_overlays = True
        
        img = keras.utils.load_img("C:\image.png", target_size = (68, 120 , 3), grayscale = False)
        img = keras.utils.img_to_array(img)
        #[0-255] -> [0-1]
        img = img/255
        
        img = np.expand_dims(img, axis=0)
        
        prediction = probability_model(img)
        rating = np.argmax(prediction[0], axis=0) + 1
        print(prediction)
        print(prediction[0][0] < 0.5)
        properties.comp_rating = rating
        
        # delete temp
        os.remove("C:/image.png")
        
        keras.backend.clear_session()
        
        #calculate progress bars
        amount_1 = int((round(float(prediction[0][0]),1)*2) * 10)
        amount_2 = int((round(float(prediction[0][1]),1)*2) * 10)
        amount_3 = int((round(float(prediction[0][2]),1)*2) * 10)
        
        properties.rating_bar1 = "1| " + amount_1*"▓" + (20-amount_1)*"▒" + " " + str(round(float(prediction[0][0]), 5))
        properties.rating_bar2 = "2| " + amount_2*"▓" + (20-amount_2)*"▒" + " " + str(round(float(prediction[0][1]), 5))
        properties.rating_bar3 = "3| " + amount_3*"▓" + (20-amount_3)*"▒" + " " + str(round(float(prediction[0][2]), 5))
        
        return {'FINISHED'}

class ColorStats(bpy.types.Operator):
    bl_idname = "wm.color"
    bl_label = "a"
    
    def execute(self, context):
        scene = context.scene
        properties = scene.custom_properties
        
        # turn off overlays
        bpy.context.space_data.overlay.show_overlays = False
        bpy.context.space_data.shading.type = 'MATERIAL'
        bpy.context.space_data.shading.use_scene_world = False
        bpy.context.space_data.shading.use_scene_lights = False
        
        #render viewport
        sce = bpy.context.scene.name
        bpy.ops.render.opengl(write_still=True)
        bpy.data.images["Render Result"].save_render("C:\image.png")
        
        image_cv2 = cv2.imread("C:\image.png")
        
        #scale down image
        scale_percent = 90 # percent of original size
        width = int(image_cv2.shape[1] * scale_percent / 100)
        height = int(image_cv2.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_img = cv2.resize(image_cv2, dim, interpolation = cv2.INTER_AREA)
        
        #[bgr] -> [rgb]
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        rows,cols,_ = resized_img.shape
        
        hsv_palette = rgb_to_hsv(np.array(resized_img)/255)
        
        hues_12 = np.zeros(12)
        
        for i in range(rows):
            for ii in range(cols):
                #print(hsv_palette)
                pix_hue = int(hsv_palette[i,ii][0]*360)
                hues_12[int(pix_hue/30)] += 1
        
        #generate bars
        len_bar = 30
        print(hues_12)
        print(hues_12/(width*height))
        print(hues_12/(width*height)*len_bar)
        
        properties.color_1 = "▓"*int(hues_12[0]/(width*height)*len_bar)
        properties.color_1 += ((len_bar - (len(properties.color_1)))*'▒')
        properties.color_2 = "▓"*int(hues_12[1]/(width*height)*len_bar)
        properties.color_2 += ((len_bar - (len(properties.color_2)))*'▒')
        properties.color_3 = "▓"*int(hues_12[2]/(width*height)*len_bar)
        properties.color_3 += ((len_bar - (len(properties.color_3)))*'▒')
        properties.color_4 = "▓"*int(hues_12[3]/(width*height)*len_bar)
        properties.color_4 += ((len_bar - (len(properties.color_4)))*'▒')
        properties.color_5 = "▓"*int(hues_12[4]/(width*height)*len_bar)
        properties.color_5 += ((len_bar - (len(properties.color_5)))*'▒')
        properties.color_6 = "▓"*int(hues_12[5]/(width*height)*len_bar)
        properties.color_6 += ((len_bar - (len(properties.color_6)))*'▒')
        properties.color_7 = "▓"*int(hues_12[6]/(width*height)*len_bar)
        properties.color_7 += ((len_bar - (len(properties.color_7)))*'▒')
        properties.color_8 = "▓"*int(hues_12[7]/(width*height)*len_bar)
        properties.color_8 += ((len_bar - (len(properties.color_8)))*'▒')
        properties.color_9 = "▓"*int(hues_12[8]/(width*height)*len_bar)
        properties.color_9 += ((len_bar - (len(properties.color_9)))*'▒')
        properties.color_10 = "▓"*int(hues_12[9]/(width*height)*len_bar)
        properties.color_10 += ((len_bar - (len(properties.color_10)))*'▒')
        properties.color_11 = "▓"*int(hues_12[10]/(width*height)*len_bar)
        properties.color_11 += ((len_bar - (len(properties.color_11)))*'▒')
        properties.color_12 = "▓"*int(hues_12[11]/(width*height)*len_bar)
        properties.color_12 += ((len_bar - (len(properties.color_12)))*'▒')
        
        # delete temp
        os.remove("C:/image.png")
        
        # turn on overlays
        bpy.context.space_data.overlay.show_overlays = True
        
        return {'FINISHED'}

# Generate section
class GeneratePanel(bpy.types.Panel):
    bl_label = "Generate"
    bl_idname = "PT_GeneratePanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI"

    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        properties = scene.custom_properties
        
        row = layout.row()
        row.label(text = (str(properties.nrObj) + " object(s)") + (" with ground" if properties.ground else ""))
        row = layout.row()
        
        row = layout.row()
        
        row.prop(properties, "nrObj")
        row.prop(properties, "ground", icon = "VIEW_PERSPECTIVE")
        
        row.operator(GenerateScene.bl_idname, text = "Generate", icon = "SCENE_DATA")
        
        if properties.ground:
            row = layout.row()
            row.prop(properties, "rndGroundBias", icon = "OUTLINER_DATA_LATTICE")
            row.prop(properties, "groundBias", slider = True)
        
        row = layout.row()
        
        
        row.prop(properties, 'ratings', expand=True)
        row.operator(SaveScene.bl_idname, text = "Save")
        row = layout.row()
        row.prop(properties, 'filepath_generate')
        
# Train section
class TrainPanel(bpy.types.Panel):
    bl_label = "Train"
    bl_idname = "PT_TrainPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI"
    
    def draw(self,context):
        layout = self.layout
        scene = context.scene
        properties = scene.custom_properties
        
        row = layout.row()
        row.label(text="Labelled data:")
        row.prop(properties, 'filepath_train', text="")
        row = layout.row()
        
        
        box = layout.box()
        obj = context.object
        row = box.row()
        
        row.operator(DataInfo.bl_idname, text = "Refresh", icon="FILE_REFRESH")
        row = box.row()
        row.label(text=" ")
        row.label(text="1")
        row.label(text="2")
        row.label(text="3")
        row.label(text="4")
        row.label(text="5")
        row.label(text="6")
        
        row = box.row()
        info = properties.info_noGround.split(' ')
        row.label(text="N")
        row.label(text=info[0])
        row.label(text=info[1])
        row.label(text=info[2])
        row.label(text=info[3])
        row.label(text=info[4])
        row.label(text=info[5])
        
        for i in range(3):
            row = box.row()
            row.alert = True
            if i == 0:
                info = properties.info_n1.split(' ')
            elif i == 1:
                info = properties.info_n2.split(' ')
            elif i == 2:
                info = properties.info_n3.split(' ')
            elif i == 3:
                info = properties.info_n4.split(' ')
            else:
                info = properties.info_n5.split(' ')
            row.label(text=str(i+1)+"|")
            row.label(text=info[0])
            row.label(text=info[1])
            row.label(text=info[2])
            row.label(text=info[3])
            row.label(text=info[4])
            row.label(text=info[5])
        
        
        row = box.row()
        info = properties.info_Ground.split(' ')
        row.label(text="G")
        row.label(text=info[0])
        row.label(text=info[1])
        row.label(text=info[2])
        row.label(text=info[3])
        row.label(text=info[4])
        row.label(text=info[5])
        
        for i in range(3):
            row = box.row()
            row.alert = True
            if i == 0:
                info = properties.info_g1.split(' ')
            elif i == 1:
                info = properties.info_g2.split(' ')
            elif i == 2:
                info = properties.info_g3.split(' ')
            elif i == 3:
                info = properties.info_g4.split(' ')
            else:
                info = properties.info_g5.split(' ')
            row.label(text=str(i+1)+"|")
            row.label(text=info[0])
            row.label(text=info[1])
            row.label(text=info[2])
            row.label(text=info[3])
            row.label(text=info[4])
            row.label(text=info[5])
            
        row = box.row()
        row.label(text=" ")
        row.label(text="1")
        row.label(text="2")
        row.label(text="3")
        row.label(text="4")
        row.label(text="5")
        row.label(text="6")
            
        row = layout.row()
        row.prop(properties, 'filepath_export', text="")
        row.prop(properties, 'epochs', text="Epochs")
        row = layout.row()
        row.operator(TrainNN.bl_idname, text = "Train and Export", icon="HAND")
        
# Evaluate section
class EvaluatePanel(bpy.types.Panel):
    bl_label = "Evaluate"
    bl_idname = "PT_EvaluatePanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        properties = scene.custom_properties
        
        row = layout.row()
        row.prop(properties, 'filepath_import', text="Trained model")
        row.operator(UseNN.bl_idname, text = "Evaluate", icon="TRACKER_DATA")
        row = layout.row()
        row.label(text="Compositional Rating: " + str(properties.comp_rating))
        
        row = layout.row()
        box = layout.box()
        row = box.row()
        row.label(text = properties.rating_bar1)
        row = box.row()
        row.label(text = properties.rating_bar2)
        row = box.row()
        row.label(text = properties.rating_bar3)

class ColorPanel(bpy.types.Panel):
    bl_label = "Color Info"
    bl_idname = "PT_ColorPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        properties = scene.custom_properties
        
        #custom icons for colors
        img_1 = bpy.data.images.load(os.path.join(bpy.path.abspath("//icons"), '0.png'))
        img_2 = bpy.data.images.load(os.path.join(bpy.path.abspath("//icons"), '30.png'))
        img_3 = bpy.data.images.load(os.path.join(bpy.path.abspath("//icons"), '60.png'))
        img_4 = bpy.data.images.load(os.path.join(bpy.path.abspath("//icons"), '90.png'))
        img_5 = bpy.data.images.load(os.path.join(bpy.path.abspath("//icons"), '120.png'))
        img_6 = bpy.data.images.load(os.path.join(bpy.path.abspath("//icons"), '150.png'))
        img_7 = bpy.data.images.load(os.path.join(bpy.path.abspath("//icons"), '180.png'))
        img_8 = bpy.data.images.load(os.path.join(bpy.path.abspath("//icons"), '210.png'))
        img_9 = bpy.data.images.load(os.path.join(bpy.path.abspath("//icons"), '240.png'))
        img_10 = bpy.data.images.load(os.path.join(bpy.path.abspath("//icons"), '270.png'))
        img_11 = bpy.data.images.load(os.path.join(bpy.path.abspath("//icons"), '300.png'))
        img_12 = bpy.data.images.load(os.path.join(bpy.path.abspath("//icons"), '330.png'))
        
        icon_1 = self.layout.icon(bpy.data.images['0.png'])
        icon_2 = self.layout.icon(bpy.data.images['30.png'])
        icon_3 = self.layout.icon(bpy.data.images['60.png'])
        icon_4 = self.layout.icon(bpy.data.images['90.png'])
        icon_5 = self.layout.icon(bpy.data.images['120.png'])
        icon_6 = self.layout.icon(bpy.data.images['150.png'])
        icon_7 = self.layout.icon(bpy.data.images['180.png'])
        icon_8 = self.layout.icon(bpy.data.images['210.png'])
        icon_9 = self.layout.icon(bpy.data.images['240.png'])
        icon_10 = self.layout.icon(bpy.data.images['270.png'])
        icon_11 = self.layout.icon(bpy.data.images['300.png'])
        icon_12 = self.layout.icon(bpy.data.images['330.png'])
        
        row = layout.row()
        row.operator(ColorStats.bl_idname, text = "Update", icon="FILE_REFRESH")
        row = layout.row()
        row.alignment = 'CENTER'
        row.label(text="Color Distribution")
        row = layout.row()
        row = layout.row()
        row.label(text=properties.color_1, icon_value=icon_1)
        row = layout.row()
        row.label(text=properties.color_2, icon_value=icon_2)
        row = layout.row()
        row.label(text=properties.color_3, icon_value=icon_3)
        row = layout.row()
        row.label(text=properties.color_4, icon_value=icon_4)
        row = layout.row()
        row.label(text=properties.color_5, icon_value=icon_5)
        row = layout.row()
        row.label(text=properties.color_6, icon_value=icon_6)
        row = layout.row()
        row.label(text=properties.color_7, icon_value=icon_7)
        row = layout.row()
        row.label(text=properties.color_8, icon_value=icon_8)
        row = layout.row()
        row.label(text=properties.color_9, icon_value=icon_9)
        row = layout.row()
        row.label(text=properties.color_10, icon_value=icon_10)
        row = layout.row()
        row.label(text=properties.color_11, icon_value=icon_11)
        row = layout.row()
        row.label(text=properties.color_12, icon_value=icon_12)
        row = layout.row()
        row = layout.row()
        row = layout.row()
        
        row = layout.row()
        row = layout.row()
        row.alignment = 'CENTER'
        row.label(text="Color Harmony")
        row = layout.row()
        row.alert = True
        row.label(text="▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ Monochromatic")
        row = layout.row()
        row.alert = True
        row.label(text="▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ Complementary")
        row = layout.row()
        row.alert = True
        row.label(text="▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ Split Complementary")
        row = layout.row()
        row.alert = True
        row.label(text="▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ Triad")
        row = layout.row()
        row.alert = True
        row.label(text="▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ Square")
        row = layout.row()
        row.alert = True
        row.label(text="▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ Rectangular")
        row = layout.row()
        row.alert = True
        row.label(text="▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ Analogous")
        row = layout.row()
        row.alert = True
        row.label(text="▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ Other")
        
        
# ------------------------------------------------------------------------
#     Registration
# ------------------------------------------------------------------------

classes = (
    GenerateScene,
    SaveScene,
    TrainNN,
    UseNN,
    GeneratePanel,
    TrainPanel,
    Properties,
    EvaluatePanel,
    DataInfo,
    ColorStats,
    ColorPanel
)

import bpy.utils.previews
icons_dict = bpy.utils.previews.new()
# this will work for addons 
icons_dir = os.path.join(os.path.dirname(__file__), "icons")
# but it won't give you usefull path when you opened a file in text editor and hit run.
# this will work in that case:
script_path = bpy.context.space_data.text.filepath
icons_dir = os.path.join(os.path.dirname(script_path), "icons")

icons_dict.load("custom_icon", os.path.abspath(os.path.join(icons_dir, "run.png")), 'IMAGE')

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    
    bpy.types.Scene.custom_properties = PointerProperty(type=Properties)
    

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    
    del bpy.types.Scene.custom_properties
    
    
if __name__ == "__main__":
    register()