/*
 * Copyright (C) 2024 Shubham Panchal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.shubham0204.smollm

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SmolLMInstrumentedTest {
    private val modelPath = "/data/local/tmp/smollm2-1.7b-instruct-q4_k_m.gguf"
    // private val systemPrompt = ""
    private val query = "How are you?"

    @Test
    suspend fun testSmolLM() {
        val smolLM = SmolLM()
        smolLM.create(modelPath)
        val responseFlow = smolLM.getResponse(query)
        responseFlow.collect {
            println(it)
        }
        smolLM.close()
    }
}
