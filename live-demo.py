import cv2 as cv
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import ops as utils_ops
import six
import collections
from PIL import Image
# import matplotlib
# from matplotlib import pyplot as plt

# initialization function that loads model and labels
def init():
    global detection_graph
    global category_index
    
    # load labeling information
    path_to_labels = 'data/object-detection.pbtxt'
    category_index = label_map_util.create_categories_from_labelmap(path_to_labels, use_display_name=True)
    if (type(category_index) != 'dict'):
        category_index = { i+1: category_index[i] for i in range(len(category_index))}
    # load frozen model
    path_to_frozen_graph = 'savedModel_191016/frozen_inference_graph.pb'
    detection_graph = loadFrozenModel(path_to_frozen_graph)
    
    # matplotlib.use('TkAgg')


# helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# load a frozen tensorflow model into memory
def loadFrozenModel(path_to_frozen_graph):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def generate_element(class_name, left, right, top, bottom):
    style = ' style={{width:' + str(right - left) + \
            ', height:' + str(bottom - top) + \
            ', position: "absolute"'+ \
            ', top:' + str(top) + \
            ', left:' + str(left) + \
            '}}'
    if class_name == 'Button':
        element = '\t\t<input type="button" value="Button" ' + style + '/>'
        pass
    elif class_name == 'CheckBox':
        element = '\t\t<input type="checkbox" value="" ' + style + '/>CheckBox'
        pass
    elif class_name == 'ComboBox':
        element =   '\t\t<select' + style + '> \
                        <option value="Value1"></option> \
                    </select>'
        pass
    elif class_name == 'Heading':
        element = '\t\t<h1 ' + style + '>Heading</h1>'
        pass
    elif class_name == 'Image':
        element = '\t\t<img src={"url_to_image"} alt="Image" ' + style + '/>'
        pass
    elif class_name == 'Label':
        element = '\t\t<label' + style + '>Label</label>'
        pass
    elif class_name == 'Link':
        element = '\t\t<a href="link_url" ' + style + '>Link</a>'
        pass
    elif class_name == 'Paragraph':
        element = '\t\t<p' + style + '></p>'
        pass
    elif class_name == 'RadioButton':
        element = '\t\t<input type="radio" value="" ' + style + '/>RadioBox'
        pass
    elif class_name == 'TextBox':
        element = '<\t\tinput type="text" ' + style + '/>'
        pass
    else:
        element = ''
        pass
    return element


def generate_html(image, output_dict, use_normalized_coordinates=True, 
                  min_score_thresh=.5, 
                  instance_masks=None, 
                  line_thickness=4):
    # initialize variables
    box_to_param_map = collections.defaultdict(list)
    boxes   = output_dict['detection_boxes']
    classes = output_dict['detection_classes']
    scores  = output_dict['detection_scores']
    global category_index
    
    # result html
    result_html = '''
import React from 'react';import './App.css';
function App() {
    return ( 
        <div>'''
    
    # some extra settings
    possible_classes = six.viewkeys(category_index)
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    
    for i in range(boxes.shape[0]):
        if scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
        else:
            continue
        if classes[i] in possible_classes:
            class_name = category_index[classes[i]]['name']
        else :
            class_name = 'N/A'
    
        box_to_param_map[box].append(class_name)
        box_to_param_map[box].append(scores[i])
        
        im_width, im_height = image.size
        
        ymin, xmin, ymax, xmax = box
        if use_normalized_coordinates:
            (left, right, top, bottom) = (xmin*im_width, xmax*im_width, ymin*im_height, ymax*im_height)
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
            
        element = generate_element(class_name, left, right, top, bottom)
        result_html += '\n\t' + element
        
    result_html += '''
        </div>);
    }
export default App;
    '''
    
    res_file = open('./preview/src/App.js', 'w')
    res_file.write(result_html)
    res_file.close()



if __name__ == "__main__":
    webcam = cv.VideoCapture(0)
    if not webcam.isOpened():
        print("Cannot open camera")
        exit()

    global category_index
    global detection_graph
    init()

    while True:
        # Capture frame-by-frame
        ret, frame = webcam.read()

        # if frame is read correctly, ret is True
        if not ret:
            print("Cannot receive frame (stream end?). Exiting ...")
            break
        
        # frame to numpy array
        image_np = frame
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        # detect the objects in the single image
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
        
        # draw bounding boxes on the detected objects
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
        
        # todo: generate html code here
        generate_html(image_np, output_dict)
        
        # display the resulting frame
        cv.imshow("Live Image Recognition", image_np)
        
        key = cv.waitKey(20)
        if key == 27 or key == ord('q'): # exit on ESC or q
            break
        if cv.getWindowProperty('Live Image Recognition', cv.WND_PROP_VISIBLE) < 1:
            break

    webcam.release()
    cv.destroyAllWindows()

