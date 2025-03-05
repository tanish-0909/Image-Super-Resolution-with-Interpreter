package com.example.repnet

import android.annotation.SuppressLint
import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import org.tensorflow.lite.DataType
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MyTFliteInterpreter internal constructor(
    private val assetManager: AssetManager,
    private val useGpu: Boolean,
    private val context: Context
){
    var interpreter: Interpreter? = null
    private var options: Interpreter.Options? = null
    private val executorService: ExecutorService = Executors.newCachedThreadPool()
    private var inputImageWidth: Int = 0
    private var inputImageHeight: Int = 0
    private var modelInputSize: Int = 0
    private lateinit var outputShape: IntArray


    private val MODEL_NAME = "repnet.tflite"

    init{
        init()
    }

    companion object {
        private const val TAG = "debugging"
        private const val FLOAT_TYPE_SIZE = 4
    }

    @SuppressLint("DefaultLocale")
    @Throws(IOException::class)
    private fun init() {
        options = Interpreter.Options()
        val model = loadModelFile(context.assets, MODEL_NAME)
        val interpreter = Interpreter(model, options!!)

        val inputShape = interpreter.getInputTensor(0).shape()
        outputShape = interpreter.getOutputTensor(0).shape()
        Log.d(TAG, String.format("input image shape required = %d, %d, %d, %d", inputShape[0], inputShape[1], inputShape[2], inputShape[3]))

        Toast.makeText(
            context,
            String.format("input image shape required = %d, %d, %d, %d", inputShape[0], inputShape[1], inputShape[2], inputShape[3]),
            Toast.LENGTH_LONG
        ).show()

        Toast.makeText(
            context,
            String.format("output image shape required = %d, %d, %d, %d", outputShape[0], outputShape[1], outputShape[2], outputShape[3]),
            Toast.LENGTH_LONG
        ).show()

        inputImageWidth = inputShape[3]  // Adjusting for (batch, channels, height, width)
        inputImageHeight = inputShape[2]
        modelInputSize = FLOAT_TYPE_SIZE * inputShape[1] * inputImageWidth * inputImageHeight

        this.interpreter = interpreter
        Log.d(TAG, "Initialized TFLite interpreter.")
    }

    fun close() {
        executorService.execute {
            interpreter?.close()
            Log.d(TAG, "Closed TFLite interpreter.")
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager, filename: String): ByteBuffer {
        val fileDescriptor = assetManager.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        Log.d(TAG, "Loaded tflite model")
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    fun run(a: Any?, b: Any?) {
        interpreter!!.run(a, b)
    }

    fun prepareInputTensor(bitmap_lr: Bitmap): TensorBuffer {

        //allocate memory to create the ByteBuffer
        val buffer = ByteBuffer.allocateDirect(modelInputSize).order(ByteOrder.nativeOrder())

        //make an array named "pixels" which will contain the rgb values of the image
        val pixels = IntArray(inputImageWidth * inputImageHeight)
        bitmap_lr.getPixels(pixels, 0, bitmap_lr.width, 0, 0, bitmap_lr.width, bitmap_lr.height)


        //bitmap is stored as ARGB => ((shift right by 16) & 0xFF) gives R
        //bitmap is stored as ARGB => ((shift right by 8) & 0xFF) gives G
        //bitmap is stored as ARGB => ((shift right by 0) & 0xFF) gives B
        //let's load the values into buffer one by one

        //since my model comes form pytorch, it expects input of form (1, 3, 720, 1280) hence my buffer is loaded like
        // R0 R1 R2 R3 ... G0 G1 G2... B0 B1 B2 ...
        //If my model were from tensorflow and needed input of form (1, 720, 1280, 3), my buffer would have to be loaded in the form:
        //R0 G0 B0 R1 G1 B1 R2 G2 B2 ....

        for (pixelValue in pixels) {
            buffer.putFloat((pixelValue shr 16 and 0xFF) / 255.0f) // loading all pixels from Red channel
        }

        for (pixelValue in pixels) {
            buffer.putFloat((pixelValue shr 8 and 0xFF) / 255.0f)  // loading all pixels from Green channel
        }

        for (pixelValue in pixels) {
            buffer.putFloat((pixelValue and 0xFF) / 255.0f)       // loading all pixels from Blue channel
        }

        buffer.rewind() // resets the pointer of buffer to the beginning so that it can be used again correctly.

        return TensorBuffer.createFixedSize(intArrayOf(1, 3, inputImageHeight, inputImageWidth), DataType.FLOAT32).apply {
            loadBuffer(buffer) // bytebuffer to TensorBuffer (a fast memory type which lies outside the scope of java)
        }
    }

    fun prepareOutputTensor(): TensorBuffer {
        return TensorBuffer.createFixedSize(intArrayOf(outputShape[0], outputShape[1], outputShape[2], outputShape[3]), DataType.FLOAT32)
    }

    fun tensorToImage(myBuffer: TensorBuffer): Bitmap {
        val buffer = myBuffer.buffer    //TensorBuffer to ByterBuffer
        buffer.rewind()     //start at the start

        val height = outputShape[2]
        val width = outputShape[3]

        // Prepare an array to store pixel values
        val pixels = IntArray(width * height)

        // Allocate space for channels
        val rChannel = FloatArray(width * height)
        val gChannel = FloatArray(width * height)
        val bChannel = FloatArray(width * height)

        // Read pixel values from ByteBuffer (NCHW format)
        for (i in 0 until width * height) {
            rChannel[i] = buffer.float // Read Red channel first
        }
        for (i in 0 until width * height) {
            gChannel[i] = buffer.float // Read Green channel next
        }
        for (i in 0 until width * height) {
            bChannel[i] = buffer.float // Read Blue channel last
        }

        // Convert to ARGB format (pixel by pixel)
        for (i in 0 until width * height) {
            val r = (rChannel[i] * 255.0f).toInt().coerceIn(0, 255)
            val g = (gChannel[i] * 255.0f).toInt().coerceIn(0, 255)
            val b = (bChannel[i] * 255.0f).toInt().coerceIn(0, 255)
            val a = 255 // Fully opaque

            // Store pixel in ARGB format
            pixels[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }

        // Create Bitmap from pixel array
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)

        Log.d(TAG, String.format("output image size %d %d",bitmap.width, bitmap.height ))

        return bitmap
    }
}