import ffmpeg
import static_ffmpeg


static_ffmpeg.add_paths() # download ffmpeg if needed

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

    def streamFrames(self, process):
        for frame in self.frames:
            process.stdin.write(frame)
        process.stdin.close()
        process.wait()

    def write(self, filename):
        '''Write frames to gif or mp4 using FFMPEG. 
        
        Args:
            filename: name of output file. File ending must be ".gif" or ".mp4".

        '''

        frameStream = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', framerate=self.fps, s='{}x{}'.format(self.width, self.height))
        if '.gif' in filename:
            paletteFile = 'palette.png'
            paletteProcess = (
                frameStream
                .output(paletteFile, filter_complex='palettegen=stats_mode=diff')
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            self.streamFrames(paletteProcess)
            
            animationProcess = (
                frameStream
                .output(filename, i=paletteFile, filter_complex='paletteuse')
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        elif '.mp4' in filename:
            animationProcess = (
                frameStream
                .output(filename, pix_fmt='yuv420p', vcodec='h264')
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        else:
            raise Exception('FFMPEGwriter expects ".gif" or ".mp4" in filename')

        self.streamFrames(animationProcess)