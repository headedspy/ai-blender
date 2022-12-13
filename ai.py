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

class LearningRateReducerCb(callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.99
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)

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
        max = 10
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
        default = "0 0 0 0 0 0 0 0 0 0"
    )
    
    info_Ground: StringProperty(
        name =  "info_ground",
        default = "0 0 0 0 0 0 0 0 0 0"
    )
    
    info_n1: StringProperty(
        name = "info_n1",
        default = "1| 0   0   0   0   0   0   0   0   0  0"
    )
    info_n2: StringProperty(
        name = "info_n2",
        default = "2| 0   0   0   0   0   0   0   0   0  0"
    )
    info_n3: StringProperty(
        name = "info_n3",
        default = "3| 0   0   0   0   0   0   0   0   0  0"
    )
    info_g1: StringProperty(
        name = "info_g1",
        default = "1| 0   0   0   0   0   0   0   0   0  0"
    )
    info_g2: StringProperty(
        name = "info_g2",
        default = "2| 0   0   0   0   0   0   0   0   0  0"
    )
    info_g3: StringProperty(
        name = "info_g3",
        default = "3| 0   0   0   0   0   0   0   0   0  0"
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
        n1 = n2 = n3 = n4 = n5 = n6 = n7 = n8 = n9 = n10 = 0
        g1 = g2 = g3 = g4 = g5 = g6 = g7 = g8 = g9 = g10 = 0
        
        i_n1_1 = i_n2_1 = i_n3_1 = i_n4_1 = i_n5_1 = i_n6_1 = i_n7_1 = i_n8_1 = i_n9_1 = i_n10_1 = 0
        i_n1_2 = i_n2_2 = i_n3_2 = i_n4_2 = i_n5_2 = i_n6_2 = i_n7_2 = i_n8_2 = i_n9_2 = i_n10_2 = 0
        i_n1_3 = i_n2_3 = i_n3_3 = i_n4_3 = i_n5_3 = i_n6_3 = i_n7_3 = i_n8_3 = i_n9_3 = i_n10_3 = 0
        
        i_g1_1 = i_g2_1 = i_g3_1 = i_g4_1 = i_g5_1 = i_g6_1 = i_g7_1 = i_g8_1 = i_g9_1 = i_g10_1 = 0
        i_g1_2 = i_g2_2 = i_g3_2 = i_g4_2 = i_g5_2 = i_g6_2 = i_g7_2 = i_g8_2 = i_g9_2 = i_g10_2 = 0
        i_g1_3 = i_g2_3 = i_g3_3 = i_g4_3 = i_g5_3 = i_g6_3 = i_g7_3 = i_g8_3 = i_g9_3 = i_g10_3 = 0
        
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
                elif filename[3] == '7':
                    g7 = g7 + 1
                    if filename[0] == '1':
                        i_g7_1 = i_g7_1 + 1
                    if filename[0] == '2':
                        i_g7_2 = i_g7_2 + 1
                    if filename[0] == '3':
                        i_g7_3 = i_g7_3 + 1
                elif filename[3] == '8':
                    g8 = g8 + 1
                    if filename[0] == '1':
                        i_g8_1 = i_g8_1 + 1
                    if filename[0] == '2':
                        i_g8_2 = i_g8_2 + 1
                    if filename[0] == '3':
                        i_g8_3 = i_g8_3 + 1
                elif filename[3] == '9':
                    g9 = g9 + 1
                    if filename[0] == '1':
                        i_g9_1 = i_g9_1 + 1
                    if filename[0] == '2':
                        i_g9_2 = i_g9_2 + 1
                    if filename[0] == '3':
                        i_g9_3 = i_g9_3 + 1
                elif filename[3] == '0':
                    g10 = g10 + 1
                    if filename[0] == '1':
                        i_g10_1 = i_g10_1 + 1
                    if filename[0] == '2':
                        i_g10_2 = i_g10_2 + 1
                    if filename[0] == '3':
                        i_g10_3 = i_g10_3 + 1
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
                elif filename[3] == '7':
                    n7 = n7 + 1
                    if filename[0] == '1':
                        i_n7_1 = i_n7_1 + 1
                    if filename[0] == '2':
                        i_n7_2 = i_n7_2 + 1
                    if filename[0] == '3':
                        i_n7_3 = i_n7_3 + 1
                elif filename[3] == '8':
                    n8 = n8 + 1
                    if filename[0] == '1':
                        i_n8_1 = i_n8_1 + 1
                    if filename[0] == '2':
                        i_n8_2 = i_n8_2 + 1
                    if filename[0] == '3':
                        i_n8_3 = i_n8_3 + 1
                elif filename[3] == '9':
                    n9 = n9 + 1
                    if filename[0] == '1':
                        i_n9_1 = i_n9_1 + 1
                    if filename[0] == '2':
                        i_n9_2 = i_n9_2 + 1
                    if filename[0] == '3':
                        i_n9_3 = i_n9_3 + 1
                elif filename[3] == '0':
                    n10 = n10 + 1
                    if filename[0] == '1':
                        i_n10_1 = i_n10_1 + 1
                    if filename[0] == '2':
                        i_n10_2 = i_n10_2 + 1
                    if filename[0] == '3':
                        i_n10_3 = i_n10_3 + 1
                
                properties.info_Ground = str(g1) + " " + str(g2) + " " + str(g3) + " " + str(g4) + " " + str(g5) + " " + str(g6) + " " + str(g7) + " " + str(g8) + " " + str(g9) + " " + str(g10)
                properties.info_noGround = str(n1) + " " + str(n2) + " " + str(n3) + " " + str(n4) + " " + str(n5) + " " + str(n6) + " " + str(n7) + " " + str(n8) + " " + str(n9) + " " + str(n10)
                
                properties.info_n1 = str(i_n1_1) + " " + str(i_n2_1) + " " + str(i_n3_1) + " " + str(i_n4_1) + " " + str(i_n5_1) + " " + str(i_n6_1) + " " + str(i_n7_1) + " " + str(i_n8_1) + " " + str(i_n9_1) + " " + str(i_n10_1)
                properties.info_n2 = str(i_n1_2) + " " + str(i_n2_2) + " " + str(i_n3_2) + " " + str(i_n4_2) + " " + str(i_n5_2) + " " + str(i_n6_2) + " " + str(i_n7_2) + " " + str(i_n8_2) + " " + str(i_n9_2) + " " + str(i_n10_2)
                properties.info_n3 = str(i_n1_3) + " " + str(i_n2_3) + " " + str(i_n3_3) + " " + str(i_n4_3) + " " + str(i_n5_3) + " " + str(i_n6_3) + " " + str(i_n7_3) + " " + str(i_n8_3) + " " + str(i_n9_3) + " " + str(i_n10_3)

                properties.info_g1 = str(i_g1_1) + " " + str(i_g2_1) + " " + str(i_g3_1) + " " + str(i_g4_1) + " " + str(i_g5_1) + " " + str(i_g6_1) + " " + str(i_g7_1) + " " + str(i_g8_1) + " " + str(i_g9_1) + " " + str(i_g10_1)
                properties.info_g2 = str(i_g1_2) + " " + str(i_g2_2) + " " + str(i_g3_2) + " " + str(i_g4_2) + " " + str(i_g5_2) + " " + str(i_g6_2) + " " + str(i_g7_2) + " " + str(i_g8_2) + " " + str(i_g9_2) + " " + str(i_g10_2)
                properties.info_g3 = str(i_g1_3) + " " + str(i_g2_3) + " " + str(i_g3_3) + " " + str(i_g4_3) + " " + str(i_g5_3) + " " + str(i_g6_3) + " " + str(i_g7_3) + " " + str(i_g8_3) + " " + str(i_g9_3) + " " + str(i_g10_3)

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
        
        #model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(68,120,3)))
        #model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        #model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        #model.add(keras.layers.Dropout(0.25))
        #model.add(keras.layers.Flatten())
        #model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        #model.add(keras.layers.Dropout(0.25))
        #model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        #model.add(keras.layers.Dropout(0.5))
        
        
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(68,120,3)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
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
        properties.comp_rating = rating
        
        # delete temp
        os.remove("C:/image.png")
        
        keras.backend.clear_session()
        
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
        row.label(text="7")
        row.label(text="8")
        row.label(text="9")
        row.label(text="10")
        
        row = box.row()
        info = properties.info_noGround.split(' ')
        row.label(text="N")
        row.label(text=info[0])
        row.label(text=info[1])
        row.label(text=info[2])
        row.label(text=info[3])
        row.label(text=info[4])
        row.label(text=info[5])
        row.label(text=info[6])
        row.label(text=info[7])
        row.label(text=info[8])
        row.label(text=info[9])
        
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
            row.label(text=info[6])
            row.label(text=info[7])
            row.label(text=info[8])
            row.label(text=info[9])
        
        
        row = box.row()
        info = properties.info_Ground.split(' ')
        row.label(text="G")
        row.label(text=info[0])
        row.label(text=info[1])
        row.label(text=info[2])
        row.label(text=info[3])
        row.label(text=info[4])
        row.label(text=info[5])
        row.label(text=info[6])
        row.label(text=info[7])
        row.label(text=info[8])
        row.label(text=info[9])
        
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
            row.label(text=info[6])
            row.label(text=info[7])
            row.label(text=info[8])
            row.label(text=info[9])
            
        row = box.row()
        row.label(text=" ")
        row.label(text="1")
        row.label(text="2")
        row.label(text="3")
        row.label(text="4")
        row.label(text="5")
        row.label(text="6")
        row.label(text="7")
        row.label(text="8")
        row.label(text="9")
        row.label(text="10")
            
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
    DataInfo
)

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