import rospy

#TODO define what a draw request (srv) is and a draw response (msg) is


def handle_service_request(req):
    print("Received request: %s" % req.data)
    #TODO upon recieving a draw request call the code to actually draw the X or line
    response = "Received: " + str(req.data)
    return DRAW_RESPONSE(True, response)

def draw_service():
    print("START DRAW_SERVICE")
    rospy.init_node('draw_service_node')
    print("draw_service inited")
    s = rospy.Service('draw_service', DRAW_REQUEST, handle_service_request)
    rospy.spin()

if __name__ == "__main__":
    draw_service()
