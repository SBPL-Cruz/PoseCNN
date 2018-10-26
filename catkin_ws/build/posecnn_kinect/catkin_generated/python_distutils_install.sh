#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
    DESTDIR_ARG="--root=$DESTDIR"
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/src/posecnn_kinect"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/install/lib/python2.7/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/install/lib/python2.7/dist-packages:/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/build/lib/python2.7/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/build" \
    "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/venv/bin/python" \
    "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/src/posecnn_kinect/setup.py" \
    build --build-base "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/build/posecnn_kinect" \
    install \
    $DESTDIR_ARG \
    --install-layout=deb --prefix="/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/install" --install-scripts="/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/install/bin"
