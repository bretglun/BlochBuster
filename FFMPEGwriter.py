import subprocess

class FFMPEGwriter:
    '''Writer class that can add matplotlib figures as frames and then write gif or mp4 using FFMPEG. '''
    
    def __init__(self, fps=15):
        self.fps = fps
        self.frames = []
        self.width = None
        self.height = None

    def addFrame(self, fig):
        '''Add matplotlib figure to frame list. 
        
        Args:
            fig: matplotlib figure.

        '''

        if not (self.width or self.height):
            self.width, self.height = fig.canvas.get_width_height()
        self.frames.append(fig.canvas.tostring_rgb()) # extract the image as an RGB string

    def write(self, filename):
        '''Write frames to gif or mp4 using FFMPEG. 
        
        Args:
            filename: name of output file. File ending must be ".gif" or ".mp4".

        '''

        if '.gif' in filename:
            paletteCmd = ('ffmpeg', 
                '-s', '{}x{}'.format(self.width, self.height), 
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-i', '-', 
                '-filter_complex', 'palettegen=stats_mode=diff', '-y', 'palette.png')
            paletteProcess = subprocess.Popen(paletteCmd, stdin=subprocess.PIPE)
            for frame in self.frames:
                paletteProcess.stdin.write(frame) # write frame to GIF palette
            paletteProcess.communicate() # Create palette
            animationCmd = ('ffmpeg', 
                '-y', # overwrite output file
                '-r', str(self.fps), # frame rate
                '-s', '{}x{}'.format(self.width, self.height), # size of image string
                '-pix_fmt', 'rgb24', # input format
                '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                '-i', 'palette.png', '-filter_complex', 'paletteuse',
                '-vframes', str(len(self.frames)), # number of frames
                filename) # file name
        elif '.mp4' in filename:
            animationCmd = ('ffmpeg', 
                '-y', # overwrite output file
                '-r', str(self.fps), # frame rate
                '-s', '{}x{}'.format(self.width, self.height), # size of image string
                '-pix_fmt', 'rgb24', # input format
                '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                '-vcodec', 'h264', # output encoding
                '-pix_fmt' ,'yuv420p', # required for some media players
                '-vframes', str(len(self.frames)), # number of frames
                filename) # file name
        else:
            raise Exception('FFMPEGwriter expects ".gif" or ".mp4" in filename')
        animationProcess = subprocess.Popen(animationCmd, stdin=subprocess.PIPE)                    
        for frame in self.frames:
            animationProcess.stdin.write(frame) # write frame to animation
        animationProcess.communicate() # Create animation