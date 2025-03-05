package com.example.repnet

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.os.storage.StorageManager
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var captureBtn: Button
    private lateinit var improveBtn: Button
    private lateinit var myImagePreview: ImageView
    private lateinit var myOutputImage: ImageView
    private lateinit var bitmap: Bitmap
    private lateinit var msgTV: TextView
    private lateinit var msgTV2: TextView

    var width: Int = 0
    var height: Int = 0
    lateinit var bitmap_lr: Bitmap
    lateinit var bitmap_sr: Bitmap

    val TAG = "debugging"
    val useGpu = false

    lateinit var myContext: Context

    @RequiresApi(Build.VERSION_CODES.R)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        captureBtn = findViewById(R.id.captureBtn)
        improveBtn = findViewById(R.id.improveBtn)
        myImagePreview = findViewById(R.id.myImagePreview)
        myOutputImage = findViewById(R.id.myOutputImage)

        myContext = this

        // button click listener
        improveBtn.setOnClickListener { v: View? ->
            // Start Resolution when low resolution image is set
            Log.d(TAG, "Resolution Start!")
            try {
                runSR()
                Log.d(TAG, "success")
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }

        captureBtn.setOnClickListener{
            var intent: Intent = Intent()           //This is what helps to open the gallery.
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            startActivityForResult(intent, 100)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if(requestCode == 100){
            val uri = data?.data        //uri = uniform resource identifier
            if(uri != null) {
                bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                var x = 0
                var y = 0
                bitmap_lr = Bitmap.createBitmap(bitmap, x, y, 1280, 720)
                myImagePreview.setImageBitmap(bitmap_lr)
                Log.d("debugging", "Bitmap successfully imported / captured")
            } else {
                Toast.makeText(this, "Failed to get image!", Toast.LENGTH_SHORT).show()
            }
        }
    }

    @SuppressLint("DefaultLocale")
    @RequiresApi(Build.VERSION_CODES.R)
    @Throws(IOException::class)
    private fun runSR() {

        val srModel: MyTFliteInterpreter = MyTFliteInterpreter(assets, useGpu, myContext)

        // Prepare image by TensorImage
        val inputTensor: TensorBuffer = srModel.prepareInputTensor(bitmap_lr)
        val outputTensor: TensorBuffer = srModel.prepareOutputTensor()

        // Run the interpreter
        val startTime = System.currentTimeMillis()
        srModel.run(inputTensor.buffer, outputTensor.buffer)

        val time = System.currentTimeMillis() - startTime
        Log.d(TAG, String.format("Spent time: %dms", time))

        // Show the result
        bitmap_sr = srModel.tensorToImage(outputTensor)
        myOutputImage.setImageBitmap(bitmap_sr)

        srModel.close()
        saveImage(bitmap_sr)

        msgTV2 = findViewById(R.id.idTVMsg2)

        msgTV2.text = String.format("time taken = %dms", time)
    }

    @RequiresApi(Build.VERSION_CODES.R)
    fun saveImage(bitmap: Bitmap) {
        val storageManager = getSystemService(STORAGE_SERVICE) as StorageManager
        val storageVolume = storageManager.storageVolumes[0] // internal Storage
        val fileImage = File(storageVolume.directory!!.path + "/Download/" + System.currentTimeMillis() + ".jpeg")

        val byteArrayOutputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream)
        val bytesArray = byteArrayOutputStream.toByteArray()
        try {
            val fileOutputStream = FileOutputStream(fileImage)
            fileOutputStream.write(bytesArray)
            fileOutputStream.close()
            Log.d(TAG, "Image saved successfully")
        } catch (e: Exception) {
            throw RuntimeException(e)
        }
    }

}