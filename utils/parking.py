import cv2
import numpy as np
from copy import deepcopy

def whRatio(w, h):

    if h / w > 0.7:
        return 1
    else:
        return (1 + 0.2 * (2 ** (1 - abs(h/w - 1))))


def plot_lines(cv2_img, slots, max_slot, vertices):

    for i in range(max_slot):
        if slots[f"slot_{str(i).zfill(2)}"] == True:
            color = (0, 0, 255)
        else :
            color = (0, 255, 0)
        temp_list = vertices[i][0:8]
        point = np.array([temp_list[0:2], temp_list[2:4], temp_list[4:6], temp_list[6:8]], dtype=np.int32)
        cv2_img = cv2.polylines(cv2_img, [point], True, color, 2)


def plot_card(cv2_img, c, conf):

    tl = 1
    tf = 1
    label = 'car:' + f'{conf:.2f}'
    c_1, c_2 = (int(c[0]), int(c[1])), (int(c[2]), int(c[3]))
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    cv2.rectangle(cv2_img, c_1, c_2, (255,0,0), thickness=tl, lineType=cv2.LINE_AA)

    c_2 = c_1[0] + t_size[0], c_1[1] - t_size[1] - 3
    cv2.rectangle(cv2_img, c_1, c_2, (255,0,0), -1, cv2.LINE_AA)  # filled
    cv2.putText(cv2_img, label, (c_1[0], c_1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def update_buffer(slots_buffer):
    for i in range(len(slots_buffer)-1, 0, -1):
        for j in range(len(slots_buffer[0])):
            slots_buffer[i][f'slot_{str(j).zfill(2)}'] = slots_buffer[i-1][f'slot_{str(j).zfill(2)}']


def check_buffer(slots_buffer):
    slots_dict = [0] * len(slots_buffer[0])
    slots = {}
    
    for i in range(len(slots_buffer)):
        for j in range(len(slots_buffer[0])):
            if slots_buffer[i][f'slot_{str(j).zfill(2)}'] == True:
                slots_dict[j] += 1
    
    for i in range(len(slots_buffer[0])):
        if slots_dict[i] >= len(slots_buffer)//2:
            slots[f'slot_{str(i).zfill(2)}'] = True
        else:
            slots[f'slot_{str(i).zfill(2)}'] = False
    
    return slots


def readVertices(vertices_meta):
    
    with open(vertices_meta, 'r') as file:
        lines = file.readlines()

    array_2d = []
    max_slot = 0
    for line in lines:
        numbers = line.strip().split(' ')
        row = [int(num) for num in numbers]
        array_2d.append(row)
        max_slot = max_slot + 1

    return(array_2d, max_slot)


def readMaskVertices(vertices_meta):
    with open(vertices_meta, 'r') as file:
        lines = file.readline()
    numbers = list(map(int, lines.strip().split(' ')))
    array = [[numbers[i], numbers[i+1]] for i in range(0, len(numbers), 2)]
    return(array)


def masking(frame, pts):
    result = frame
    result = cv2.fillPoly(result, [np.array(pts, np.int32)], 1)
    return result


def isPointInside(point_x, point_y, vertices):
    #              x[1]y[1]              x[2]y[2]       
    #                o- - - - - - - - - - -o            
    #               /                     /             
    #              /                     /              
    #             /                     /               
    #            /                     /                
    #           /                     /                 
    #          /                     /                  
    #         o- - - - - - - - - - -o                   
    #      x[0]y[0]             x[3]y[3]                
    #                                               
    # Unpack the vertices into x-y pairs
    x = vertices[0::2]
    y = vertices[1::2]

    if point_x < min(x) or point_x > max(x) or point_y < min(y) or point_y > max(y):
        return False
    
    try:
        left_grad = (y[1] - y[0]) / (x[1] - x[0])
    except ZeroDivisionError:
        left_grad = float('inf')
    try:
        right_grad = (y[2] - y[3]) / (x[2] - x[3])
    except ZeroDivisionError:
        right_grad = float('inf')
    
    left_linear = y[0] - left_grad * x[0]
    right_linear = y[3] - right_grad * x[3]
    
    if left_grad == float('inf'):
        x_min = x[0]
    else:
        x_min = (point_y - left_linear) / left_grad
    if right_grad == float('inf'):
        x_max = x[3]
    else:
        x_max = (point_y - right_linear) / right_grad

    if point_x < x_min or point_x > x_max:
        return False

    top_grad = (y[2] - y[1]) / (x[2] - x[1])
    bot_grad = (y[3] - y[0]) / (x[3] - x[0])
    top_linear = y[2] - top_grad * x[2]
    bot_linear = y[3] - bot_grad * x[3]
    y_min = top_grad * point_x + top_linear
    y_max = bot_grad * point_x + bot_linear

    if point_y < y_min or point_y > y_max:
        return False

    return True
