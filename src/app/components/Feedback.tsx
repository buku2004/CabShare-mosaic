"use client";
import React, { useState } from "react";
import { db } from "../constants/firebase";
import { doc, setDoc } from "firebase/firestore";

const FeedbackForm = () => {
  
  // const [showForm, setShowForm] = useState(false);

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    phone: "",
    message: "",
    consent: false,
  });

  const handleChange = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value, type } = event.target;

    if (type === "checkbox") {
      const checked = (event.target as HTMLInputElement).checked;
      setFormData({
        ...formData,
        [name]: checked,
      });
    } else {
      setFormData({
        ...formData,
        [name]: value,
      });
    }
  };

  const handleSubmit = async (e:React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      const docID = formData.name;
      const feedbackDocRef = doc(db, "feedbacks", docID);
      // Add the listing to Firestore
      await setDoc(feedbackDocRef, formData);
      alert("Listing submitted successfully!");
      setFormData({
        name: "",
        email: "",
        phone: "",
        message: "",
        consent: false,
      });
      // setShowForm(false);
    } catch (error) {
      console.error("Error adding document: ", error);
      alert(`Failed to submit feedback: ${(error as Error).message}`);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-[#E8C547] p-8 rounded-lg shadow-md w-full max-w text-[#2E2E2E]"
    >
      <h2 className="text-4xl text-[#2E2E2E] font-bold mb-4"> We are here to help you ride share!</h2>
      <p className="text-[#2E2E2E] mb-6">Get in touch by giving your valuable feedback</p>

      <label className="block mb-2 text-sm font-medium">
        Name *
        <input
          type="text"
          name="name"
          value={formData.name}
          onChange={handleChange}
          required
          className="w-full mt-1 p-2 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
        />
      </label>

      <label className="block mb-2 text-sm font-medium">
        Email Address *
        <input
          type="email"
          name="email"
          value={formData.email}
          onChange={handleChange}
          required
          className="w-full mt-1 p-2 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
        />
      </label>

      <label className="block mb-2 text-sm font-medium">
        Phone Number *
        <input
          type="tel"
          name="phone"
          value={formData.phone}
          onChange={handleChange}
          required
          className="w-full mt-1 p-2 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
        />
      </label>

      <label className="block mb-4 text-sm font-medium">
        Message
        <textarea
          name="message"
          value={formData.message}
          onChange={handleChange}
          className="w-full mt-1 p-2 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
          rows={4}
        ></textarea>
      </label>

      <label className="flex items-center mb-6 text-sm font-bold">
        <input
          type="checkbox"
          name="consent"
          checked={formData.consent}
          onChange={handleChange}
          required
          className="mr-2 w-4 h-4 text-green-500 border-gray-700 rounded focus:ring-green-500"
        />
        I allow this website to store my submission so they can respond to my inquiry. *
      </label>

      <button
        type="submit"
        className="w-full bg-[#2E2E2E] text-white py-2 px-4 rounded-md hover:bg-green-600 transition-colors"
      >
        Submit
      </button>
    </form>
  );
};

const Feedback = () => {
  return (
    <div className="flex flex-col lg:flex-row gap-8 p-8 bg-white text-white ">
      <div className="flex-1 flex justify-center h-[37.5rem]">
        <FeedbackForm />
      </div>
    </div>
  );
};

export default Feedback;
