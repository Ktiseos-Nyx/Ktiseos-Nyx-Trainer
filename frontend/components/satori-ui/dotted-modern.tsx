"use client";
import Link from "next/link";
import React from "react";
import { motion } from "motion/react";
import { ArrowRight, Terminal } from "lucide-react";

type MotionWrapperProps = {
  children: React.ReactNode;
};
const MotionWrapper = ({ children }: MotionWrapperProps) => {
  return (
    <motion.span
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.4,
        ease: "easeOut",
        delay: 0.3,
      }}
      className="relative inline-block overflow-hidden p-[1px]"
    >
      {children}
    </motion.span>
  );
};

const DottedModern = () => {
  return (
    <div className="flex flex-col relative h-full w-full">
      <div className="absolute inset-0 z-0 h-full w-full bg-white bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] [background-size:16px_16px] rounded-md"></div>
      <div className="mx-auto h-full flex flex-col gap-6 items-center justify-center">
        <MotionWrapper>
          <a
            href="https://github.com/ibelick/background-snippets"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex"
          >
            <span
              className="relative inline-block overflow-hidden rounded-full p-[1px] border border-slate-800 border-1"
              style={{ animationDelay: "0.3s", animationFillMode: "forwards" }}
            >
              <div
                className="
                inline-flex h-full w-full cursor-pointer items-center justify-center
                rounded-full bg-white px-3 py-1 text-xs font-medium leading-5
                text-slate-700 backdrop-blur-xl
                bg-white
              "
              >
                We are open source 🚀
                <span className="inline-flex items-center pl-1 font-semibold text-black">
                  Github
                  <ArrowRight className="pl-0.5 text-black" size={16} />
                </span>
              </div>
            </span>
          </a>
        </MotionWrapper>

        <MotionWrapper>
          <h1 className="font-display text-4xl sm:text-6xl lg:text-7xl font-medium tracking-tight leading-[1.05] text-slate-900">
            Action - oriented
            <br />
            <span className="bg-clip-text text-transparent bg-gradient-to-br from-slate-900 via-slate-600 to-slate-400">
              Modern snippets
            </span>
          </h1>
        </MotionWrapper>

        <MotionWrapper>
          <p className="text-lg text-slate-500 leading-relaxed text-center max-w-xl">
            Plug-and-play snippets-just copy, paste, and use in your next
            project. Built with Tailwind CSS and Vanilla CSS for seamless
            integration.
          </p>
        </MotionWrapper>
        <MotionWrapper>
          <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto pt-2">
            <Link
              href="/components/button"
              className="group w-full sm:w-auto px-6 py-3 bg-slate-900 hover:bg-slate-800 text-white rounded-sm font-semibold text-sm shadow-xl shadow-slate-200/50 flex items-center justify-center gap-2"
            >
              Go to Github
              <ArrowRight className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-[2px]" />
            </Link>

            <button className="w-full sm:w-auto px-6 py-3 rounded-sm font-semibold text-sm text-slate-600 border border-slate-200 hover:bg-slate-50 hover:text-slate-900 transition-all flex items-center justify-center gap-2 bg-white">
              <Terminal className="h-4 w-4" />
              Documentation
            </button>
          </div>
        </MotionWrapper>
      </div>
    </div>
  );
};

export default DottedModern;
