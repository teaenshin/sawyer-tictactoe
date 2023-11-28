import rospy


def handle_service_request(req):
    print("Received request: %s" % req.data)
    response = "Received: " + str(req.data)
    return DRAW_RESPONSE(True, response)

def draw_service():
    rospy.init_node('draw_service_node')
    s = rospy.Service('draw_service', DRAW_REQUEST, handle_service_request)
    rospy.spin()

if __name__ == "__main__":
    draw_service()
